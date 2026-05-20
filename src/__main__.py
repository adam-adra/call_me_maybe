from src import llm_generation
from src import Loading
import argparse
from src.state_machine.state import Output
import os
from typing import List
import json

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
    llm_generation(data, args.output)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt as e:
        print("The program ended")
