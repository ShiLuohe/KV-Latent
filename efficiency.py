
import torch

import gc
from tqdm import tqdm
from transformers import AutoTokenizer


from modelcodes.llama3 import (
    get_model,
    MLAConfig
)

from localdatasets.dsfinewebedu import get

PROMPTS = ['''Here is a recipe of a classic chicken sandwitch. Step 1:''' * 136 + "Step 2. Add boiled water"]

lora_r = 256
lora_a = 2 * lora_r

result_ttft = []
result_mspt = []

for rr_qk in [1, 2, 4, 8]:
    for rr_vo in [1, 2, 4, 8]:
        if rr_qk * rr_vo == 1:
            continue

        gc.collect()
        config = MLAConfig(rr_qk, rr_vo, lora_r, lora_a, f"./checkpoints/ablations0807/qk{rr_qk}vo{rr_vo}")
        device, tokenizer, model, orig_model = get_model(config, True, False)

        device = torch.device("cuda")
        model = model.to(device)
        model.set_training(False)

        tt_ttft = 0.
        tt_mspt = 0.
        for i in tqdm(range(5), desc=f"Testing qk{rr_qk} vo{rr_vo}"):
            responses, ttft, mspt = model.generatewithtime(tokenizer, PROMPTS, 256)
            tt_ttft += ttft
            tt_mspt += mspt
        
        result_ttft.append(tt_ttft / 5)
        result_mspt.append(tt_mspt / 5)

print(*result_ttft, sep=" & ")
print(*result_mspt, sep=" & ")
        

