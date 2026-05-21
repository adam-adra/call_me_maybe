from dataclasses import dataclass
from src.models import Loading
from typing import List, Dict
from src.vocab.load_vocab import Vocab
from llm_sdk import Small_LLM_Model
import json
from pydantic import BaseModel, ValidationError
from time import sleep
from src.vocab.load_vocab import VocabLoader
import re

@dataclass
class Literal:
    text: str


@dataclass
class Slot:
    name: str
    type: str


class Output(BaseModel):
    prompt: str
    name: str
    parameters: Dict


class StateMachine:

    def __init__(self,
                 vl: VocabLoader,
                 u_prompt: str,
                 model: Small_LLM_Model):
        self.template = list()
        self.vc = vl.vc
        self.position: int = int()
        self.state: int = 0
        self.token_id: List[int] = list()
        self.model: Small_LLM_Model = model
        self.final: Dict[int: List[int]] = dict()
        self.end = False
        self.u_prompt = u_prompt
        self.index = 0
        self.vl = vl
        self.params_items: List = list()

    def template_builder(self,
                         fn_name: str, 
                         data: Loading
                         ) -> None:
        """Given a known function name and 
        its parameter schema, produce the exact sequence of fixed tokens
        and slot markers that the final JSON should follow

        Args:
            fn_name (str): function name
            data (Loading): functions data

        Returns:
            List: seqence of fixed tokens
        """
        schema = [fn.parameters  for fn in data.list_function 
                  if str(fn.name) == fn_name]
        self.params_items: List = list(schema[0].items())

        first_name, first_type = self.params_items[0]
        first_piece = f'{{"name": "{fn_name}", "parameters": {{'
        if first_type.type == "string":
            first_piece += f'"{first_name}": "'
        else:
            first_piece += f'"{first_name}": '
        self.template.append(
            Literal(first_piece)
        )
        self.template.append(Slot(
            name=first_name, type=first_type.type
        ))
        for param_name, param_info in self.params_items[1:]:
            param_type = param_info.type
            if param_type == "string":
                self.template.append(
                    Literal(f', "{param_name}": "')
                )
            else:
                self.template.append(
                    Literal(f', "{param_name}": ')
                )
            self.template.append(
                Slot(
                    name=param_name,
                    type=param_type
                )
            )
        last_type = self.params_items[-1][1].type
        if last_type == "string":
            self.template.append(Literal('}}'))
        else:
            self.template.append(Literal('}}'))


    def get_valid_token_ids(self) -> List[int]:
        id: List[int] = list()

        if self.final[self.position] and self.final[self.position][-1] == self.vc.decimal_tokens:
            self.state = 2

        if self.state == 0:
            id.extend(self.vc.number_start_token)
            self.state = 1

        elif self.state == 1:
            id.extend(self.vc.digit_tokens)
            if self.template[self.index].type.lower() == "number":
                id.append(self.vc.decimal_tokens)
            id.append(self.terminator())

        elif self.state == 2:
            id.extend(self.vc.digit_tokens)
            id.append(self.terminator())
        return id


    def advance(self) -> None:
        segment = self.template[self.index]
        if (segment.type.lower() == "number") or (segment.type.lower() == "integer"):
            while not self.is_done():
                self.mask()
        elif (segment.type.lower() == "string"):
            while not self.is_done():
                self.str_mask()
        self.end = False


    def mask(self):
        logits = self.model.get_logits_from_input_ids(
            self.token_id
        )
        nxt_tk = self.get_valid_token_ids()


        current = self.final[self.position]
        if len(current) >= 10:
            self.end = True
            return


        logits_choosen = [float('-inf') if index not in nxt_tk else log
                          for index, log in enumerate(logits) ]
        next_token = logits_choosen.index(max(logits_choosen))
        if next_token == self.terminator():
            self.end = True
            return
        self.token_id.append(next_token)
        self.final[
            self.position].append(next_token)

    def str_mask(self):
        generated = self.final[self.position]
        if generated and generated[-1] == self.vc.quote_token:
            self.end = True
            return

        n = len(generated)

        # Detect copy-from-prompt mode: if everything generated so far appears
        # verbatim in the user request, the model is transcribing a value (e.g.
        # source_string) and needs room to finish. Otherwise it is synthesising
        # a fresh value (regex, replacement) and must stop quickly.
        if n > 0:
            partial = self.model.decode(generated).strip()
            is_copying = bool(partial) and partial in self.u_prompt.prompt
        else:
            is_copying = False

        # Repetition detector — only applied to synthesised values; copied
        # text legitimately repeats tokens (e.g. "another cat").
        if not is_copying:
            for k in (1, 2, 3, 4):
                if n >= 2 * k and generated[-k:] == generated[-2 * k:-k]:
                    generated.append(self.vc.quote_token)
                    self.token_id.append(self.vc.quote_token)
                    self.end = True
                    return

        hard_limit = 40 if is_copying else 16
        if n >= hard_limit:
            generated.append(self.vc.quote_token)
            self.token_id.append(self.vc.quote_token)
            self.end = True
            return

        original_logits = self.model.get_logits_from_input_ids(self.token_id)
        logits = [float('-inf')] * len(original_logits)

        for token_str, token_id in self.vl.str_to_id.items():
            if token_str != '"' and '"' not in token_str:
                logits[token_id] = original_logits[token_id]

        if is_copying:
            boost = max(0, n - 10) * 1.0
        else:
            boost = max(0, n - 2) * 4.0
        logits[self.vc.quote_token] = original_logits[self.vc.quote_token] + boost

        next_token = logits.index(max(logits))
        self.token_id.append(next_token)
        generated.append(next_token)

    def is_done(self) -> bool:
        if self.end:
            self.state = 0
            return self.end
        return False


    def terminator(self) -> int:
        nx_seg = self.template[self.index + 1].text[0]
        if nx_seg == ",":
            return self.vc.comma_token
        elif nx_seg == '"':
            return self.vc.quote_token
        elif nx_seg == "}":
            return self.vc.close_brace_token

    def extract_quoted_spans(self) -> List[str]:
        """Pull out single and double quoted strings from the user prompt."""
        return re.findall(r"['\"]([^'\"]+)['\"]", self.u_prompt.prompt)

    def current_segment(self) -> Output:
        base_instruction = (
            "Write only the exact value needed. "
            "Be concise and stop immediately when the value is complete. "
            "Do not repeat, pad, or continue after the value ends. "
            "Use the shortest pattern that correctly matches; never repeat the pattern."
        )
        examples = (
            "Examples of correct, minimal outputs:\n"
            'Request: Replace digits in "abc 42 def" with X\n'
            '{"name": "fn_substitute_string_with_regex", "parameters": {"source_string": "abc 42 def", "regex": "[0-9]+", "replacement": "X"}}\n'
            "Request: Replace vowels in 'hello world' with -\n"
            '{"name": "fn_substitute_string_with_regex", "parameters": {"source_string": "hello world", "regex": "[aeiou]", "replacement": "-"}}\n'
            "Request: Replace 'foo' with 'bar' in 'foo and foo again'\n"
            '{"name": "fn_substitute_string_with_regex", "parameters": {"source_string": "foo and foo again", "regex": "foo", "replacement": "bar"}}\n'
        )
        base_prompt = base_instruction + "\n\n" + examples + "\n" + f'User request: {self.u_prompt.prompt}\n'
        prompt = base_prompt  # tracks the growing JSON context

        for self.index in range(len(self.template)):

            if isinstance(self.template[self.index], Literal):
                prompt += self.template[self.index].text
                self.token_id = self.model.encode(prompt)[0].tolist()

            elif isinstance(self.template[self.index], Slot):
                slot_name = self.template[self.index].name
                # Rebuild token_id with a slot-specific prompt
                # candidates = self.extract_quoted_spans()
                slot_prompt = (
                        f"You are filling the parameter \"{slot_name}\".\n"
                    ) + prompt
                self.token_id = self.model.encode(slot_prompt)[0].tolist()

                self.final[self.position] = list()
                self.advance()
                self.position += 1

        return self.build_answer()


    def build_answer(self) -> Output | None:
        """
        Buildin the json format of the generated parameter and
        the function name and the prompt of the user
        """
        answer_loop = str()
        index = 0
        data = dict()
        result = None
        try:
            self.streaming(self.u_prompt.prompt)
            print()
            for sg in self.template:
                if isinstance(sg, Literal):
                    answer_loop += sg.text
                    self.streaming(sg.text)
                if isinstance(sg, Slot):
                    param = self.model.decode(self.final[index])
                    self.streaming(param)
                    answer_loop += param
                    index += 1
            print("\n")
            data.update({"prompt": self.u_prompt.prompt})
            for key, value in json.loads(answer_loop).items():
                data[key] = value
            result = Output.model_validate(data)
            return result
        except Exception as e:
            print("[Error]: ", e, end="\n\n")
        except ValidationError as e:
            print("[Error]: ", e, end="\n\n")
        return None


    def streaming(self, text: str):
        for c in text:
            print(c, flush=True, end="")
            sleep(0.02)