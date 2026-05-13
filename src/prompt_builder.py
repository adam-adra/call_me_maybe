from src.models import FunctionDefinition, FunctionCalling
from typing import List


class Prompt:
    """
        This class is for generating prompt
    """

    def __init__(self,
                 functions: List[FunctionDefinition],
                 test_prompts: List[FunctionCalling]
                 ):
        """The class Prompt to create a system prompt for
            the LLM
                function definition
        Args:
            functions (List[FunctionDefinition]):
        """
        self.base_prompt: str = self.__build_base_prompt(functions)
        self.prompts: List[str] = list()
        self.__generate_full(test_prompts)

    def __build_base_prompt(self,
                            functions:
                            List[FunctionDefinition]
                            ) -> str:
        """
        This method to build a base prompt based on
            the functions definition we have
        Args:
            functions (List[FunctionDefinition]):
                function definition

        Returns:
            str: base prompt
        """
        prompt: str
        function_name: str
        function_description: str
        function_result: str

        prompt = ("You are a function selection assistant.\n"
                  "Available functions:")

        for function in functions:
            function_name = function.name
            function_description = function.description
            function_result = f"- {function_name}: {function_description}"
            prompt = "\n".join([prompt, function_result])

        return prompt

    def build_prompt(self, user_prompt: str) -> str:
        """
        This method is to generate a prompt
        based on the base_prompt using the
        user prompt to generate a LLM prompt

        Args:
            user_prompt (str): user prompt

        Returns:
            str: LLM prompt
        """
        prompt: str

        prompt = "\n".join([self.base_prompt,
                            (f"\nUser request: \"{user_prompt}\"\n"
                             " Answer with only the function name."
                             "\nWhich function should be called?")])
        self.prompts.append(prompt)
        return prompt

    def __generate_full(self,
                        test_prompts:
                        List[FunctionCalling]) -> None:
        """
        generate the prompts list for each test prompt
        Args:
            test_prompts (List[FunctionCalling]):
                the test prompts
        """
        for prompt in test_prompts:
            self.build_prompt(prompt.prompt)
