
import torch

from tqdm import tqdm
import random
import time

from modelcodes.llama2 import (
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
    del inputs["token_type_ids"]
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

rr_qk = 1
rr_vo = 1
lora_r = 256
lora_a = 2 * lora_r

# config = MLAConfig(rr_qk, rr_vo, 256, 512, f"./checkpoints/ablations0807/qk{rr_qk}vo{rr_vo}")
# config = MLAConfig(rr_qk, rr_vo, lora_r, lora_a, f"./checkpoints/ablations0807/qk8vo8")
# config = MLAConfig(rr_qk, rr_vo, 256, 512, f"./checkpoints/distill0813b/")
config = MLAConfig(rr_qk, rr_vo, lora_r, lora_a, f"./checkpoints/distill0813b")
_, tokenizer, model, orig_model = get_model(config, False, True)

device = torch.device("cuda")
# model.load_state_dict(torch.load("./checkpoints/240812a_dist/model_state.pt"))

model = model.to(device)
# orig_model = orig_model.to(device)
model.set_training(False)

file = open("./attentionisallyouneed.txt")
txt = file.read()

t_ttft = 0
t_mspt = 0

for i in tqdm(range(15)):
    line = "I take pride in my content so feel free to have a look around my channel and subscribe if you enjoy! Thanks for watching! :)"
    question = "I take pride in my content so feel free"
    tokens = tokenizer([txt, line, question]).input_ids

    # print(len(tokens), len(tokens[0]), len(tokens[1]), len(tokens[2]))

    max_length = 3840 - len(tokens[1]) - len(tokens[2])
    pos = random.randint(0, max_length - 1)
    tokens = tokens[0][0:pos] + tokens[1] + tokens[0][pos:max_length] + tokens[2]
    tokens = torch.LongTensor([tokens])
    # print(tokens.shape)

    # responses, ttft, mspt = model.generatewithtoken(tokenizer, tokens, 32)
    responses, ttft, mspt = generatewithtoken(model, tokenizer, tokens, 32)
    print(ttft, mspt)
    # print(responses, ttft, mspt)
    t_ttft += ttft
    t_mspt += mspt

print(t_ttft / 15, t_mspt / 15)

