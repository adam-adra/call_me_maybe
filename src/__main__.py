from src import llm_generation
from src import Loading
import argparse


def main() -> None:
    """
    Main entry point for running the Call Me Maybe function calling system.
    Parses command-line arguments for function definitions, prompts,
    and output paths,then executes the constrained LLM generation pipeline.
    """
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

    parser.add_argument(
        "--model",
        type=str,
        default="Qwen/Qwen3-0.6B",
        help="HuggingFace model identifier to use"
    )

    args = parser.parse_args()
    data = Loading(args.functions_definition, args.input)
    llm_generation(data, args.output, args.model)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("The program ended")
    except Exception as e:
        print(e)
