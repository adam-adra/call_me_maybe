from llm_sdk import Small_LLM_Model
from typing import List, Dict, Any
from src.vocab.load_vocab import VocabLoader
from src.models import Loading
from src.state_machine.state import StateMachine
from src.prompt_builder import Prompt
import os
import json


def llm_generation(data: Loading,
                   path: str,
                   model_name: str) -> None:
    """
    Executes the main constrained decoding pipeline for the local LLM.
    Runs logit masking for function name selection, then builds and
    runs the State Machine to securely generate schema-compliant
    function parameters. Saves outputs to file.

    Args:
        data (Loading): Loaded schema and prompt definitions.
        path (str): Target file path to write results.
        model_name (str): HuggingFace model identifier.
    """
    prompt = Prompt(data.list_function, data.list_prompts)
    model: Small_LLM_Model = Small_LLM_Model(model_name=model_name)
    respond: List[int] = list()
    text_respond: str = str()
    list_output: List[Dict[str, Any]] = list()

    # filling the str to id and the id to str
    VocabLoader.read_vocab(model.get_path_to_vocab_file())
    VocabLoader.tokenize_function_name(
        model, data
    )
    VocabLoader.load_vc()
    # build a tree based on the token id we got from tokenize_fn_name
    VocabLoader.build_trie()

    for pm_nb in range(len(prompt.prompts)):
        respond = list()
        token_id: List[int] = list()
        print("Prompt Number: ", pm_nb + 1, end="\n")
        pr = prompt.prompts[pm_nb]  # to change this later
        encoded_result = model.encode(pr)
        token_id = encoded_result[0].tolist()

        while True:
            logits = model.get_logits_from_input_ids(token_id)
            masked_logits = VocabLoader.mask_logits(logits,
                                                    respond)
            next_token = masked_logits.index(max(masked_logits))
            if masked_logits[next_token] == float('-inf'):
                break
            respond.append(next_token)
            token_id.append(next_token)

        text_respond = model.decode(respond)

        sm = StateMachine(
            VocabLoader,
            data.list_prompts[pm_nb],
            model
        )
        sm.template_builder(
            fn_name=text_respond,
            data=data
        )
        result = sm.current_segment()
        if result:
            list_output.append(result.model_dump())
        write_output(list_output, path)
        print("=" * 50)


def write_output(
    results: List[Dict[str, Any]],
    path: str
) -> None:
    """
    Writes the list of validated output dictionaries to the target JSON file,
    ensuring all target folders are created successfully.

    Args:
        results (List[Dict[str, Any]]): List of validated JSON object maps.
        path (str): File path where outputs should be saved.
    """
    try:
        folder = os.path.dirname(path)
        if folder:
            os.makedirs(folder, exist_ok=True)
        with open(path, "w+") as f:
            json.dump(results, f, indent=4)
    except Exception as e:
        print(e)
