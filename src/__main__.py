from src import llm_generation
from src import Loading
# from src import Prompt
import argparse
from src.state_machine.state import Output
import os
from typing import List
import json


def write_output(results: List[Output],
                path: str) -> None:
    try:
        folder = os.path.dirname(path)
        if folder:
            os.makedirs(folder, exist_ok=True)
        with open(path, "w+") as f:
            json.dump(results, f, indent=4)
    except Exception as e:
        print(e)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Taking the function definition and input and output")

    parser.add_argument("--functions_definition",
                        type=str,
                        default="data/input/functions_definition.json",
                        help="Path to function definitions JSON")

    parser.add_argument("--input",
                        type=str,
                        default="data/input/function_calling_tests.json",
                        help="Path to the prompts JSON")

    parser.add_argument("--output",
                        type=str,
                        default="data/output/function_calling_results.json",
                        help="Path to output JSON")

    args = parser.parse_args()
    data = Loading(args.functions_definition, args.input)
    results = llm_generation(data)
    write_output(results, args.output)





if __name__ == "__main__":
    main()
