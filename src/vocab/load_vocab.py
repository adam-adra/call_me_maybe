import json
from typing import Dict, List
from llm_sdk import Small_LLM_Model
from src.models import Loading
from dataclasses import dataclass

class TrieNode:

    def __init__(self) -> None:
        self.children: Dict[int, TrieNode] = dict()
        self.end = False


class Vocab:


    def __init__(self):
        self.number_start_token: List[int] = list()
        self.digit_tokens: List[int] = list()
        self.decimal_tokens: int = None
        self.quote_token: int = None
        self.open_brace_token: int = None
        self.close_brace_token: int = None
        self.colon_token: int = None
        self.comma_token: int = None
        self.space_token: int = None


class VocabLoader:

    str_to_id: Dict[str, int] = dict()
    id_to_str: Dict[int, str] = dict()
    fn_token: Dict[str, List[int]] = dict()
    space: str = "Ġ"
    root = TrieNode()
    vc = Vocab()

    @classmethod
    def read_vocab(cls, path: str) -> None:
        """
        this is a class method that takes the path of the
        vocab file and turn them into 2 dict one is str
        to id and the other id to str

        Args:
            path (str): path for vocab file
        """
        try:
            with open(path, 'r') as f:
                cls.str_to_id = json.load(f)
            cls.id_to_str = {i: s
                             for s, i in
                             cls.str_to_id.items()}
        except Exception as e:
            print("[ERROR]:", e)

    @classmethod
    def load_vc(cls):
        """
        Load the attribute of the object of the class 
        Vocab
        """
        attributes: Dict = {
                            "decimal_tokens": ".",
                            "quote_token": '"',
                            "open_brace_token": "{",
                            "close_brace_token": "}",
                            "colon_token": ":",
                            "comma_token": ",",
                            "space_token": cls.space 
                           }

        for d in [str(d) for d in range(0, 10)]:
            cls.vc.digit_tokens.append(
                cls.str_to_id.get(d)
            )
            cls.vc.number_start_token.append(
                cls.str_to_id.get(d)
            )
        cls.vc.number_start_token.append(
            cls.str_to_id.get("-")
        )
        for attribute, char in attributes.items():
            value = cls.str_to_id[char]
            setattr(cls.vc, attribute, value)

    @classmethod
    def tokenize_function_name(cls,
                               model: Small_LLM_Model,
                               data: Loading) -> None:
        """This method is to get the token id of
        each function name we have and place them
        inside a dict

        Args:
            model (Small_LLM_Model): the Small_LLM_Model
            data (Loading): Where we have
                the List of functions
        """
        prefix: str = '{"name": "'
        prefix_len: int = len(
            model.encode(prefix)[0].tolist())
        tokens: List[int]

        # for function in data.list_function:
        #     params = list(function.parameters.keys())
        #     full_string = (prefix + f'{function.name}"' 
        #                    + ", " + '"parameters": {')
        #     for param in params:
        #         full_string += f' "{param}": '
        #         if param != params[-1]:
        #             full_string += ","
        #     full_string += " } }"

        for function in data.list_function:
            full_string = prefix + function.name
            tokens = model.encode(full_string)[0].tolist()
            cls.fn_token.update(
                {function.name: tokens[prefix_len:]}
            )

    @classmethod
    def build_trie(cls) -> None:
        """
        This method is to generate a trie (prefix tree)
        with all the token id of the defined functions
        """
        for tokens in cls.fn_token.values():
            node = cls.root
            for token in tokens:
                if token not in node.children:
                    node.children[token] = TrieNode()
                node = node.children[token]
        node.end = True

    @classmethod
    def get_valid_next_ids(cls,
                           id_chosen: List[int]
                           ) -> List[int]:
        """This method is to get the potentially next
        token id

        Args:
            id_chosen (List[int]):
            List of already chosen id

        Returns:
            Optional[List[int]]: the potential
            next id
        """
        node = cls.root
        for id in id_chosen:
            if id not in node.children.keys():
                return []
            node = node.children[id]
        # if node.end:
        #     word = [cls.id_to_str[id] for id in id_chosen]
        #     return "".join(word)
        return list(node.children.keys())

    @classmethod
    def valid_token_ids(
                        cls,
                        is_valid: int,
                        chosen: List[int]
                        ) -> bool:
        """check if this token id is in valid
        Args:
            is_valid (int): token id
            chosen (list[int]): chosen ids
        Returns:
            bool: is valid
        """
        valid = cls.get_valid_next_ids(chosen)
        return is_valid in valid

    @classmethod
    def mask_logits(cls,
                    logits_list: List[float],
                    chosen: List[int]
                    ) -> List[float]:
        masked = [
            float('-inf') if not cls.valid_token_ids(index, chosen) else fl
            for index, fl in enumerate(logits_list)
        ]
        return masked
