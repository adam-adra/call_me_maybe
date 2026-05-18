from llm_sdk import Small_LLM_Model
from typing import List, Optional
from src.vocab.load_vocab import VocabLoader
from src.models import Loading
from src.state_machine.state import StateMachine
from src.prompt_builder import Prompt
from src.state_machine.state import Output
import os
import json

def llm_generation(data: Loading) -> str:

    prompt = Prompt(data.list_function, data.list_prompts)
    model = Small_LLM_Model()
    respond: List[int] = list()
    text_respond: str = str()
    list_output: List[Output] = list()
    
    pm_nb = 0

    prompt = prompt.prompts[pm_nb] # to change this later


    token_id = model.encode(prompt)[0].tolist()
    VocabLoader.read_vocab(model.get_path_to_vocab_file()) # filling the str to id and the id to str
    VocabLoader.tokenize_function_name(
        model, data
    )   
    VocabLoader.load_vc()
    VocabLoader.build_trie() # build a tree based on the token id we got from tokenize_fn_name
    
   
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


    sm = StateMachine(VocabLoader.vc,
                      data.list_prompts[pm_nb],
                      model)
    sm.template_builder(fn_name=text_respond,
                                 data=data)

    list_output.append(sm.current_segment().model_dump())
    return list_output


