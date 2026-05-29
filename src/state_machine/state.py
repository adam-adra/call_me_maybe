from dataclasses import dataclass
from src.models import Loading, FunctionCalling
from typing import List, Dict, Union, Any, Optional
from llm_sdk import Small_LLM_Model
import json
from pydantic import BaseModel, ValidationError
from time import sleep
from src.vocab.load_vocab import VocabLoader
import re


@dataclass
class Literal:
    """Represents a fixed literal token sequence in the output template.

    Attributes:
        text (str): The literal text to emit into the generated JSON.
    """
    text: str


@dataclass
class Slot:
    """Represents a parameter slot in the output template.

    Attributes:
        name (str): Parameter name.
        type (str): Parameter type (e.g., "string", "number").
    """
    name: str
    type: str


class Output(BaseModel):
    """Pydantic model describing a validated output object.

    Fields:
        prompt (str): The original user prompt.
        name (str): Selected function name.
        parameters (Dict[str, Any]): Generated parameters for the function.
    """
    prompt: str
    name: str
    parameters: Dict[str, Any]


class StateMachine:
    """State machine that guides constrained generation of function parameters.

    The state machine is provided with a token vocabulary helper, a user
    prompt, and a lightweight LLM instance. It constructs a template
    (mix of Literal and Slot segments) for the target JSON and then
    drives token-level generation constrained by the vocabulary.
    """

    def __init__(self,
                 vl: type[VocabLoader],
                 u_prompt: FunctionCalling,
                 model: Small_LLM_Model) -> None:
        self.template: List[Union[Literal, Slot]] = list()
        self.vc = vl.vc
        self.position: int = int()
        self.state: int = 0
        self.token_id: List[int] = list()
        self.model: Small_LLM_Model = model
        self.final: Dict[int, List[int]] = dict()
        self.end = False
        self.u_prompt = u_prompt
        self.index = 0
        self.vl = vl
        self.params_items: List[Any] = list()

    def template_builder(
        self,
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
        schema = [fn.parameters for fn in data.list_function
                  if str(fn.name) == fn_name]
        self.params_items = list(schema[0].items())

        first_name, first_type = self.params_items[0]
        first_piece = f'{{"name": "{fn_name}", "parameters": {{'
        if first_type.type == "string":
            first_piece += f'"{first_name}": "'
        elif first_type.type == "boolean":
            first_piece += f'"{first_name}": '
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
        """Return a list of token ids that are valid for the current slot state

        The returned ids depend on the current numeric/string parsing state
        (start, integer digits, decimal digits) and include the appropriate
        terminator token for the following template segment.
        """
        id: List[int] = list()

        pos_final = self.final.get(self.position, [])
        if pos_final and pos_final[-1] == self.vc.decimal_tokens:
            self.state = 2

        if self.state == 0:
            id.extend(self.vc.number_start_token)
            self.state = 1

        elif self.state == 1:
            id.extend(self.vc.digit_tokens)
            curr_seg = self.template[self.index]
            is_num = (
                isinstance(curr_seg, Slot)
                and curr_seg.type.lower() == "number"
            )
            if is_num:
                assert self.vc.decimal_tokens is not None
                id.append(self.vc.decimal_tokens)
            id.append(self.terminator())
        elif self.state == 2:
            id.extend(self.vc.digit_tokens)
            id.append(self.terminator())
        return id

    def advance(self) -> None:
        """Advance generation for the current template segment.

        If the current segment is a `Slot` this will repeatedly call the
        appropriate masking routine (`mask` for numbers, `str_mask` for
        strings, `mask_bool` for booleans) until the slot is complete.
        """
        segment = self.template[self.index]
        try:
            if isinstance(segment, Slot):
                seg_type = segment.type.lower()
                if seg_type in ("number", "integer"):
                    while not self.is_done():
                        self.mask()
                elif seg_type == "string":
                    while not self.is_done():
                        self.str_mask()
                elif seg_type == "boolean":
                    while not self.is_done():
                        self.mask_bool()
            self.end = False
        except Exception as e:
            print("[Error]: ", e, end="\n\n")
            self.end = True

    def mask(self) -> None:
        """Mask logits for numeric slots and select the next valid token.

        Applies vocabulary-based filtering to logits, selects the highest
        scoring valid token, appends it to the running token lists, and
        sets the `end` flag when the slot terminator or a safety limit is
        reached.
        """
        logits = self.model.get_logits_from_input_ids(self.token_id)
        nxt_tk = self.get_valid_token_ids()
        current = self.final[self.position]
        if len(current) >= 10:
            self.end = True
            return
        logits_choosen = [float('-inf') if index not in nxt_tk else log
                          for index, log in enumerate(logits)]
        next_token = logits_choosen.index(max(logits_choosen))
        if next_token == self.terminator():
            self.end = True
            return
        self.token_id.append(next_token)
        self.final[self.position].append(next_token)

    def str_mask(self) -> None:
        """Mask logits when generating string slots and select the next token.

        This method prevents repetition, enforces a quote terminator, and
        biases the model towards finishing the string slot. It also
        protects against copying long substrings directly from the user
        prompt by adjusting token boosts.
        """
        generated = self.final[self.position]
        if generated and generated[-1] == self.vc.quote_token:
            self.end = True
            return
        n = len(generated)
        if n > 0:
            partial = self.model.decode(generated).strip()
            is_copying = bool(partial) and partial in self.u_prompt.prompt
        else:
            is_copying = False
        if not is_copying:
            for k in (1, 2, 3, 4):
                if n >= 2 * k and generated[-k:] == generated[-2 * k:-k]:
                    assert self.vc.quote_token is not None
                    generated.append(self.vc.quote_token)
                    self.token_id.append(self.vc.quote_token)
                    self.end = True
                    return
        hard_limit = 40 if is_copying else 16
        if n >= hard_limit:
            assert self.vc.quote_token is not None
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
        q_token = self.vc.quote_token
        assert q_token is not None
        logits[q_token] = original_logits[q_token] + boost
        next_token = logits.index(max(logits))
        self.token_id.append(next_token)
        generated.append(next_token)

    def mask_bool(self) -> None:
        """Mask logits for boolean slots and select the next valid token.

        Restricts generation to only "true" or "false" tokens. Once one of
        these complete keywords is generated, the slot is marked as done.
        """
        generated = self.final[self.position]
        original_logits = self.model.get_logits_from_input_ids(self.token_id)
        logits = [float('-inf')] * len(original_logits)

        assert self.vc.true_token is not None
        assert self.vc.false_token is not None

        logits[self.vc.true_token] = original_logits[self.vc.true_token]
        logits[self.vc.false_token] = original_logits[self.vc.false_token]

        next_token = logits.index(max(logits))
        self.token_id.append(next_token)
        generated.append(next_token)
        self.end = True

    def is_done(self) -> bool:
        """Return whether the current slot generation is finished.

        If the slot has finished, reset the numeric `state` to the start
        state for the next slot and return True; otherwise return False.
        """
        if self.end:
            self.state = 0
            return self.end
        return False

    def terminator(self) -> int:
        """Return the token id that should terminate the current slot.

        The terminator is chosen based on the first character of the next
        `Literal` segment (comma, quote, or closing brace). Raises on
        unexpected characters.
        """
        next_seg = self.template[self.index + 1]
        if not isinstance(next_seg, Literal):
            raise ValueError("Expected next segment to be a Literal")
        nx_seg = next_seg.text[0]
        if nx_seg == ",":
            assert self.vc.comma_token is not None
            return self.vc.comma_token
        elif nx_seg == '"':
            assert self.vc.quote_token is not None
            return self.vc.quote_token
        elif nx_seg == "}":
            assert self.vc.close_brace_token is not None
            return self.vc.close_brace_token
        raise ValueError(f"Unexpected next segment character: {nx_seg}")

    def extract_quoted_spans(self) -> List[str]:
        """Extract quoted substrings from the user prompt.

        Returns a list of strings that were inside single or double quotes
        in the original user prompt. Useful for seeding slot values.
        """
        return re.findall(r"['\"]([^'\"]+)['\"]", self.u_prompt.prompt)

    def current_segment(self) -> Optional[Output]:
        """Drive generation for the full template and
        return a validated Output with retry logic.

        Builds a sequence of instructions and iterates through the template
        (Literals and Slots), filling each slot via constrained generation.
        Returns a validated `Output` instance on success or `None` on error.

        On JSON parsing failure, retries once with a modified prompt prefix.
        """
        attempt = 0
        max_attempts = 2

        while attempt < max_attempts:
            try:
                return self._attempt_segment(attempt)
            except json.JSONDecodeError as e:
                attempt += 1
                if attempt < max_attempts:
                    print(f"[Retry {attempt}] JSON parse failed: {e}")
                    self.reset_for_retry()
                else:
                    print("[Error]: ", e, end="\n\n")
                    return None
            except (ValidationError, Exception) as e:
                print("[Error]: ", e, end="\n\n")
                return None

        return None

    def _attempt_segment(self, attempt: int = 0) -> Optional[Output]:
        """Internal method to attempt one pass through template generation.

        Args:
            attempt (int): Current attempt number (0-based) for prompt
                modification.

        Returns:
            Optional[Output]: Validated output or None on non-JSON errors.

        Raises:
            json.JSONDecodeError: When JSON parsing fails (triggers retry).
            ValidationError: When output validation fails.
        """
        base_instruction = (
            "Write only the exact value needed. "
            "Be concise and stop immediately when the value is complete. "
            "Do not repeat, pad, or continue after the value ends. "
            "Use the shortest pattern that correctly matches; "
            "never repeat the pattern."
        )

        # Modify instruction slightly on retry
        if attempt > 0:
            base_instruction = (
                "Output only the required value. Be minimal and precise. "
                "Stop as soon as the value is complete. "
                "Do not add anything extra. "
                "Use the minimal matching pattern; never duplicate."
            )

        examples = (
            "Examples of correct, minimal outputs:\n"
            'Request: Replace digits in "abc 42 def" with X\n'
            '{"name": "fn_substitute_string_with_regex", '
            '"parameters": {"source_string": "abc 42 def", '
            '"regex": "[0-9]+", "replacement": "X"}}\n'
            "Request: Replace vowels in 'hello world' with -\n"
            '{"name": "fn_substitute_string_with_regex", '
            '"parameters": {"source_string": "hello world", '
            '"regex": "[aeiou]", "replacement": "-"}}\n'
            "Request: Replace 'foo' with 'bar' in "
            "'foo and foo again'\n"
            '{"name": "fn_substitute_string_with_regex", '
            '"parameters": {"source_string": "foo and foo again", '
            '"regex": "foo", "replacement": "bar"}}\n'
        )
        base_prompt = (
            base_instruction + "\n\n" + examples + "\n"
            + f"User request: {self.u_prompt.prompt}\n"
        )
        prompt = base_prompt  # tracks the growing JSON context

        for self.index in range(len(self.template)):
            seg = self.template[self.index]
            if isinstance(seg, Literal):
                prompt += seg.text
                self.token_id = self.model.encode(prompt)[0].tolist()

            elif isinstance(seg, Slot):
                slot_name = seg.name
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

    def reset_for_retry(self) -> None:
        """Reset the state machine for a retry attempt with modified prompt.

        Clears generated tokens and parameter tracking for a fresh attempt
        with a slightly different prompt prefix.
        """
        # Clear the final generations
        self.final = {i: list() for i in range(len(self.params_items))}
        # Reset position and token tracking
        self.position = 0
        self.token_id = list()
        self.state = 0
        self.end = False

    def build_answer(self) -> Optional[Output]:
        """
        Build the json format of the generated parameter and
        the function name and the prompt of the user.

        Returns None if validation fails, or Output on success.
        Raises json.JSONDecodeError if JSON parsing fails (for retry
            handling).
        """
        answer_loop = str()
        index = 0
        data = dict()
        result = None

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
        # This may raise json.JSONDecodeError
        for key, value in json.loads(answer_loop).items():
            data[key] = value
        result = Output.model_validate(data)
        return result

    def streaming(self, text: str) -> None:
        for c in text:
            print(c, flush=True, end="")
            sleep(0.02)
