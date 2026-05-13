from llm_sdk import Small_LLM_Model
from typing import List, Optional
from src.vocab.load_vocab import VocabLoader
from src.models import Loading


def llm_generation(prompt: str,
                   max_token: int,
                   data: Loading) -> str:
    model = Small_LLM_Model()
    respond: List[int] = list()
    text_respond: str = str()
    eos_id = model._tokenizer.pad_token_id
    token_id = model.encode(prompt)[0].tolist()
    VocabLoader.read_vocab(model.get_path_to_vocab_file())
    VocabLoader.tokenize_function_name(
        model, data
    )
    VocabLoader.build_trie()

    for i in range(0, max_token):
        logits = model.get_logits_from_input_ids(token_id)
        masked_logits = VocabLoader.mask_logits(logits,
                                                respond)
        next_token = masked_logits.index(max(masked_logits))
        if masked_logits[next_token] == float('-inf'):
            break
        respond.append(next_token)
        token_id.append(next_token)
        if (next_token == eos_id):
            break
    text_respond = model.decode(respond)
    return text_respond


def extractor(text: str, functions: List[str]) -> Optional[str]:
    splited: List[str] = text.split()
    for word in splited:
        if word.lower() in functions:
            return word
    return None
