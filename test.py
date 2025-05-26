
import torch

from tqdm import tqdm
from transformers import PreTrainedTokenizer, PreTrainedModel

ANSWER2ID = {"A": 0, "B": 1, "C": 2, "D": 3, "E": 4, "F": 5, "G": 6, "H": 7, }

@torch.no_grad()
def test_ppl(model: PreTrainedModel, tokenizer: PreTrainedTokenizer, dataloader, max_entry=500):
    ppl = 0
    total = 0
    device = model.device
    for inedx, data in enumerate(tqdm(dataloader, 
                                      desc="Testing perplexity", 
                                      total=min(len(dataloader), max_entry), 
                                      leave=False)):
        if inedx >= max_entry: break
        inputs = tokenizer(data["text"], return_tensors="pt", padding="longest", truncation=True)
        total += inputs.input_ids.shape[0]
        # del inputs["token_type_ids"]
        inputs = inputs.to(device)
        outputs = model(**inputs, labels=inputs.input_ids)
        ppl += outputs.loss.item()
        # if inedx % 50 == 0:
        #     print(outputs.loss.item(), end=" ")  
    
    return ppl / total

@torch.no_grad()
def test_acc(model: PreTrainedModel, tokenizer: PreTrainedTokenizer, dataloader):
    acc = 0
    total = 0
    device = model.device
    for data in tqdm(dataloader, desc="Testing accuracy", total=len(dataloader), leave=False):
        inputs = tokenizer(data["text"], return_tensors="pt", padding="longest", truncation=True)
        # del inputs["token_type_ids"]
        inputs = inputs.to(device)
        total += len(data["text"])
        outputs = model(**inputs, labels=inputs.input_ids)
        logits = outputs.logits.cpu()
        for entry, answer in zip(logits, data["answer"]):
            opt = ANSWER2ID.get(tokenizer.decode(entry[-1].argmax(dim=-1))[0])
            if opt is None:
                total -= 1
            acc += opt == answer.item()
    
    if total == 0: return 0.
    return acc / total, total


from modelcodes.llama3 import (
    # model, 
    # tokenizer, 
    # device, 
    set_length_for_training, 
    get_total_loss, 
    clear_total_loss,
    # save_model,
    # load_model,
    # set_training,
    get_model,
    MLAConfig
)

rr_qk = 2
rr_vo = 2
lora_r = 256
lora_a = 2 * lora_r

# config = MLAConfig(rr_qk, rr_vo, 256, 512, f"./checkpoints/ablations0807/qk{rr_qk}vo{rr_vo}")
config = MLAConfig(rr_qk, rr_vo, lora_r, lora_a, f"./checkpoints/240812a_dist")
# config = MLAConfig(rr_qk, rr_vo, 256, 512, f"./checkpoints/distill0813b/")
device, tokenizer, model, orig_model = get_model(config, False, False)

from localdatasets.dsfinewebedu import get
batch_size = 1
validloader, _, _ = get(batch_size=batch_size, shuffle=True)
from localdatasets.obqa import get_dataset
obqa = get_dataset()

device = torch.device("cuda")
model.load_state_dict(torch.load("./checkpoints/240812a_dist/model_state.pt"))
model = model.to(device)
# orig_model = orig_model.to(device)
model.set_training(False)

PROMPTS = [
# '''The best thing to do on a Sunday in San Francisco is''', 
'''Here is a recipe of a classic chicken sandwitch. Step 1:''',
]

# responses, ttft, mspt = model.generatewithtime(tokenizer, PROMPTS, 256)
# print(responses, ttft, mspt)

# print(test_acc(model, tokenizer, obqa))
print(test_ppl(model, tokenizer, validloader))