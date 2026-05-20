from llm_sdk import Small_LLM_Model
from typing import List
from src.vocab.load_vocab import VocabLoader
from src.models import Loading
from src.state_machine.state import StateMachine
from src.prompt_builder import Prompt
from src.state_machine.state import Output
import os
import json


def llm_generation(data: Loading, path: str):

    prompt = Prompt(data.list_function, data.list_prompts)
    model = Small_LLM_Model()
    respond: List[int] = list()
    text_respond: str = str()
    list_output: List[Output] = list()

    VocabLoader.read_vocab(model.get_path_to_vocab_file()) # filling the str to id and the id to str
    VocabLoader.tokenize_function_name(
        model, data
    )
    VocabLoader.load_vc()    
    VocabLoader.build_trie() # build a tree based on the token id we got from tokenize_fn_name

    for pm_nb in range(len(prompt.prompts)):
        respond = list()
        token_id = list()

        pr = prompt.prompts[pm_nb] # to change this later
        token_id = model.encode(pr)[0].tolist()

        while 1:
            logits = model.get_logits_from_input_ids(token_id)
            masked_logits = VocabLoader.mask_logits(logits,
                                                    respond)
            next_token = masked_logits.index(max(masked_logits))
            if masked_logits[next_token] == float('-inf'):
                break
            respond.append(next_token)
            token_id.append(next_token)
        
        text_respond = model.decode(respond)

        sm = StateMachine(VocabLoader,
                        data.list_prompts[pm_nb],
                        model)
        sm.template_builder(fn_name=text_respond,
                                    data=data)
        result = sm.current_segment()
        if result:
            list_output.append(result.model_dump())
        write_output(list_output, path)


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