
import torch

from tqdm import tqdm
import random
import time

from modelcodes.llama3 import (
    get_model,
    MLAConfig
)

from transformers.generation.utils import DynamicCache

@torch.no_grad()
def generatewithtoken(model, tokenizer, tokens: torch.Tensor, max_new_token=256):
    key_value_cache = DynamicCache()
    device = model.model.embed_tokens.weight.device

    inputs = tokenizer("Message", return_tensors="pt", padding=True, truncation=True)
    inputs["input_ids"] = tokens
    del inputs["attention_mask"]
    inputs = inputs.to(device)

    new_tokens = []
    while len(new_tokens) < max_new_token:
        if len(new_tokens) == 0:
            time1 = time.time()
        elif len(new_tokens) == 1:
            time2 = time.time()
        outputs = model.forward(**inputs, 
                                past_key_values=key_value_cache, 
                                use_cache=True,
                                # output_attentions = True
                                )
        key_value_cache = outputs.past_key_values
        next_token = outputs.logits[..., -1:, :].argmax(dim=-1).cpu()

        new_tokens.append(next_token)

        inputs["input_ids"] = next_token.to(device)
        inputs["attention_mask"] = torch.ones_like(next_token, device=device)

    time3 = time.time()

    new_tokens = torch.cat(new_tokens, dim=-1)
    return  tokenizer.batch_decode(new_tokens), \
            (time2 - time1), \
            (time3 - time2) / max_new_token

rr_qk = 8
rr_vo = 1
lora_r = 256
lora_a = 2 * lora_r

config = MLAConfig(rr_qk, rr_vo, 256, 512, f"./checkpoints/ablations0807/qk{rr_qk}vo{rr_vo}")
_, tokenizer, model, orig_model = get_model(config, True, False)

device = torch.device("cuda")

model = model.to(device)
model.set_training(False)

prompts = ["Here is a detailed recipe of a chicken sandwich. First:", 
           "The best thing to do in San Francisco on a cloudy Sunday is", 
           "To be or not to be,", 
           "Attention is all you need."
           ]

for prompt in prompts:
    tokens = tokenizer(prompt, return_tensors="pt").input_ids
    response, ttft, mspt = generatewithtoken(model, tokenizer, tokens, 32)

    print(response)
    print(f"Time to first token: {int(ttft * 100) / 100} s. \n Milisecond per token: {int(mspt * 1000)} ms.")

