from pydantic import BaseModel, ValidationError
from typing import Literal, Dict, List
import json
import os


class VariableType(BaseModel):
    type: Literal["number", "string", "boolean"]


class FunctionDefinition(BaseModel):
    name: str
    description: str
    parameters: Dict[str, VariableType]
    returns: VariableType


class FunctionCalling(BaseModel):
    prompt: str


class Loading:

    def __init__(self,
                 function_path: str,
                 prompts_path: str):
        self.list_function = self.load_function_definition(function_path)
        self.list_prompts = self.load_test_prompts(prompts_path)

    def load_function_definition(self, path: str) -> List[FunctionDefinition]:
        """
        Load and validate function definitions from a JSON file.
        This function reads a JSON file containing a
        list of function definition objects, validates
        each entry using the FunctionDefinition Pydantic model,
        and returns the raw data if all entries are valid.

        Parameters
        ----------
        path : str
            Path to the JSON file containing function definitions.

        Returns
        -------
        Optional[List[FunctionDefinition]]
            A list of validated function definition FunctionDefinition
                if successful.
            Returns None if:
            - The file contains invalid JSON
            - Any function definition fails schema validation
            - Any unexpected error occurs

        Behavior
        --------
        - Each function definition is validated individually.
        - On validation failure, an error message is printed indicating
        which function index failed (1-based index).
        - The function exits early on the first encountered error.
        """

        try:
            validated = []
            with open(path, "r") as f:
                data = json.load(f)
            for function_index, function_definition in enumerate(data):
                validated.append(FunctionDefinition.
                                 model_validate(function_definition))
        except json.JSONDecodeError as e:
            print(e)
            exit(1)
        except ValidationError as e:
            print(
                f"[Error]: {e.errors()[0]['msg']} "
                f"in function number: {function_index + 1}"
            )
            exit(1)
        except Exception as e:
            print("[Error]:", str(e))
            exit(1)
        return validated

    def load_test_prompts(self, path: str) -> List[FunctionCalling]:
        """
        Load and validate Function Calling from a JSON file.
        This function reads a JSON file containing a list
        of Function Calling objects, validates each entry
        using the FunctionCalling Pydantic model,
        and returns the raw data if all entries are valid.

        Parameters
        ----------
        path : str
            Path to the JSON file containing function definitions.

        Returns
        -------
        Optional[List[FunctionCalling]]
            A list of validated prompts definition
            FunctionCalling if successful.
            Returns None if:
            - The file contains invalid JSON
            - Any function definition fails schema validation
            - Any unexpected error occurs

        Behavior
        --------
        - Each function definition is validated individually.
        - On validation failure, an error message is printed indicating
        which function index failed (1-based index).
        - The function exits early on the first encountered error.
        """
        try:
            validated = []
            with open(path, "r") as f:
                data = json.load(f)
            for prompt_index, prompt in enumerate(data):
                validated.append(FunctionCalling.
                                 model_validate(prompt))
        except json.JSONDecodeError as e:
            print(e)
            exit(1)
        except ValidationError as e:
            print(
                f"[Error]: {e.errors()[0]['msg']} "
                f"in function number: {prompt_index + 1}"
            )
            exit(1)
        except Exception as e:
            print("[Error]:", str(e))
            exit(1)
        return validated

    def write_results(self, results: List[FunctionDefinition],
                      path: str) -> None:
        try:
            json_data = list()
            for result in results:
                json_data.append(result.model_dump())
            folder = os.path.dirname(path)
            if folder:
                os.makedirs(folder, exist_ok=True)
            with open(path, "w+") as f:
                json.dump(json_data, f, indent=2)
        except Exception as e:
            print(f"[Error]: {e}")
            exit(1)
