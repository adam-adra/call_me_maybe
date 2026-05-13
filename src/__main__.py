from src import extractor, llm_generation
from src import Loading
from src import Prompt
import argparse


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
    data.write_results(data.list_function, args.output)

    test_prompts = Prompt(data.list_function, data.list_prompts)
    text: str = llm_generation(test_prompts.prompts[0], 50, data)

    print(text)
    print(extractor(text, ["fn_add_numbers",
                           "fn_greet",
                           "fn_reverse_string",
                           "fn_get_square_root",
                           "fn_substitute_string_with_regex"]))


if __name__ == "__main__":
    main()
