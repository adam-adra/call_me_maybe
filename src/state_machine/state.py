from dataclasses import dataclass
from src.models import Loading
from typing import List, Dict
from src.vocab.load_vocab import Vocab
from llm_sdk import Small_LLM_Model
import json
from pydantic import BaseModel, ValidationError


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
                 vc: Vocab,
                 u_prompt: str,
                 model: Small_LLM_Model):
        self.template = list()
        self.vc = vc
        self.position: int = int()
        self.state: int = 0
        self.token_id: List[int] = list()
        self.model: Small_LLM_Model = model
        self.final: Dict[int: List[int]] = dict()
        self.end = False
        self.u_prompt = u_prompt



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
        params_items: List = list(schema[0].items())

        first_name, first_type = params_items[0]
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
        for param_name, param_info in params_items[1:]:
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
        last_type = params_items[-1][1].type
        if last_type == "string":
            self.template.append(Literal('"}}'))
        else:
            self.template.append(Literal('}}'))




    def get_valid_token_ids(self, index: int) -> List[int]:
        id: List[int] = list()

        if self.final[self.position] and self.final[self.position][-1] == self.vc.decimal_tokens:
            self.state = 2

        if self.state == 0:
            id = self.vc.number_start_token
            self.state = 1

        elif self.state == 1:
            id = self.vc.digit_tokens
            id.append(self.vc.decimal_tokens)
            id.append(self.terminator(
                index
            ))

        elif self.state == 2:
            id = self.vc.digit_tokens
            id.append(self.terminator(
                index
            ))
        return id


    def advance(self, 
                index: int) -> None:
        segment = self.template[index]
        if segment.type.lower() == "number".lower():
            while not self.is_done():
                self.mask(index)
            self.end = False



    def mask(self, index: int):
        logits = self.model.get_logits_from_input_ids(
            self.token_id
        )
        nxt_tk = self.get_valid_token_ids(index)
        logits_choosen = [float('-inf') if index not in nxt_tk else log
                          for index, log in enumerate(logits) ]
        next_token = logits_choosen.index(max(logits_choosen))
        if next_token == self.terminator(index):
            self.end = True
            return
        self.token_id.append(next_token)
        self.final[
            self.position].append(next_token)



    def is_done(self) -> bool:
        if self.end:
            self.state = 0
            return self.end
        return False



    def terminator(self,
                    index: int) -> int:
        nx_seg = self.template[index + 1].text[0]
        if nx_seg == ",":
            return self.vc.comma_token
        elif nx_seg == '"':
            return self.vc.quote_token
        elif nx_seg == "}":
            return self.vc.close_brace_token



    def current_segment(self) -> Output:
        # this is where it begins
        prompt = f'{{"prompt": "{self.u_prompt.prompt}",'
        for index in range(len(self.template)):

            if isinstance(self.template[index], Literal):
                prompt += self.template[index].text
                self.token_id = (self.model.encode(prompt)[0].tolist())

            if isinstance(self.template[index], Slot):
                self.final[
                    self.position] = list()
                self.advance(index)
                self.position += 1
        
        return self.build_answer()


    def build_answer(self) -> Output:
        """
        Buildin the json format of the generated parameter and
        the function name and the prompt of the user
        """
        answer_loop = str()
        index = 0
        data = dict()
        try:
            for sg in self.template:
                if isinstance(sg, Literal):
                    answer_loop += sg.text
                if isinstance(sg, Slot):
                    param = self.model.decode(self.final[index])
                    answer_loop += param
                    index += 1        

            data.update({"prompt": self.u_prompt.prompt})
            for key, value in json.loads(answer_loop).items():
                data[key] = value
        
            return Output.model_validate(data)
        except Exception as e:
            print(e)
        except ValidationError as e:
            print(e)
        # print(data.model_dump_json(indent=4).__class__.__name__)

