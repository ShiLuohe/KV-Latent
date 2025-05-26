import torch
from transformers import LlamaForCausalLM, PreTrainedTokenizerFast

# device = torch.device("cuda")
# tokenizer = LlamaTokenizerFast.from_pretrained("models/Meta-Llama-3-8B", trust_remote_code=True)
# tokenizer.pad_token = tokenizer.eos_token
# model = LlamaForCausalLM.from_pretrained("models/Meta-Llama-3-8B", trust_remote_code=True).half().to(device)

from codes.MLAConfig import MLAConfig
from modelcodes.modeling_llama3 import (
    LlamaForCausalLMWithMLA,
    set_length_for_training, 
    get_total_loss, 
    clear_total_loss
)
# for i in range(len(model.model.layers)):
#     model.model.layers[i] = LlamaDecoderLayerWithMLA(model.model.layers[i], "qk", 2, 32, 64)
#     model.model.layers[i].set_training(True)

# def save_model(path: str):
#     if path[-1] != '/': path = path + '/'
#     for i in range(len(model.model.layers)):
#         model.model.layers[i].save_param(path, i)
# def load_model(path: str):
#     if path[-1] != '/': path = path + '/'
#     for i in range(len(model.model.layers)):
#         model.model.layers[i].load_param(path, i)

# def set_training(new_state: bool):
#     for i in range(len(model.model.layers)):
#         model.model.layers[i].set_training(new_state)

def get_model(mla_config: MLAConfig, load_model=False, return_original_model=False):
    tokenizer = PreTrainedTokenizerFast.from_pretrained("models/Llama-2-7b-hf")
    tokenizer.pad_token = tokenizer.eos_token
    model = LlamaForCausalLM.from_pretrained("models/Llama-2-7b-hf").half()
    model.config.use_cache = False
    mlamodel = LlamaForCausalLMWithMLA(model, mla_config)
    if load_model:
        mlamodel.load_model()
    mlamodel.set_training(False)
    if return_original_model:
        new_model = LlamaForCausalLM.from_pretrained("models/Llama-2-7b-hf").half()

    return None, tokenizer, mlamodel, new_model if return_original_model else None

remove_tti = True