 
import torch
from tqdm import tqdm

from modelcodes.llama2 import (
    set_length_for_training, 
    get_total_loss, 
    clear_total_loss,
    get_model,
    LlamaForCausalLMWithMLA,
    MLAConfig
)
from localdatasets.dsfinewebedu import get

# PATH_L = "./checkpoints/240801a/"
rr_qk = 2
rr_vo = 2
lr_rk = 256
lr_al = 2 * lr_rk
PATH_S = f"./checkpoints/distill0814b/"

batch_size = 1
epochs = 1
entries = 250000
log_freq = 0.0001
sav_freq = 0.1
temp = 10

config = MLAConfig(reduce_ratio_qk=rr_qk, reduce_ratio_vo=rr_vo, 
                   lora_rank=lr_rk, lora_alpha=lr_al, 
                   path=PATH_S)
device, tokenizer, model, orig_model = get_model(config, False, True)
device = torch.device("cuda")

# config = MLAConfig("qkvo", 2, 128, 256, PATH_L)
# device, tokenizer, model = get_model(config, False)

model.set_training(True)
model = model.to(device)
orig_model = orig_model.to(device)

dataloader, _, _ = get(batch_size, shuffle=True)
# dataset["train"] = dataset["test"]
# def tokenize_function(examples):
#     outputs = tokenizer(examples["text"], truncation=True, return_tensors="pt")
#     return outputs

# # with accelerator.main_process_first():
# tokenized_datasets = dataset.map(
#     tokenize_function,
#     batched=True,
#     # remove_columns=["text"],
# )
# print(dataset["test"]["text"][0])
# input()

final_steps = min(len(dataloader) * epochs, entries)
# final_steps = 5
log_gap = int(log_freq * final_steps)
sav_gap = int(sav_freq * final_steps)
# log_gap = 1


optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=2e-7, eps=2e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=final_steps)

# model, optimizer, training_dataloader = accelerator.prepare(
#     model, optimizer, dataloader
# )

losses = []

KL_div_loss = torch.nn.KLDivLoss(reduction="batchmean")

for epoch in range(epochs):
    for index, data in tqdm(enumerate(dataloader), 
                            desc=f"Fitting MLA {epoch + 1} / {epochs}", 
                            total=final_steps,
                            # disable = not accelerator.is_local_main_process
                            ):
        
        if index >= final_steps: break

        inputs = tokenizer(data["text"], return_tensors="pt", padding=True, truncation=True)
        del inputs["token_type_ids"]
        # set_length_for_training(inputs.attention_mask.sum(dim=-1))
        inputs = inputs.to(device)

        with torch.no_grad():
            t_logits = orig_model(**inputs).logits
            t_logits = torch.nn.functional.softmax(t_logits / temp, dim=-1)

        optimizer.zero_grad()
        p_logits = model(**inputs).logits
        p_logits = torch.nn.functional.log_softmax(p_logits / temp, dim=-1)
        
        # print(t_logits[..., 15, 128:144], t_logits[..., 15, 128:144].max(dim=-1), 
        #       p_logits[..., 15, 128:144], p_logits[..., 15, 128:144].max(dim=-1), sep="\n")
        # input()
        
        loss = KL_div_loss(p_logits, t_logits)
        loss.backward()

        # print(loss)

        optimizer.step()
        scheduler.step()

        if index % log_gap == 0:
            losses.append(loss.detach().item())
            with open(PATH_S + "losses.txt", "+a") as file:
                print(loss, file=file)
        if index % sav_gap == 0:
            model.save_model(PATH_S)

model.save_model(PATH_S)
tensored_loss = torch.Tensor(losses)
torch.save(tensored_loss, PATH_S + "losses.pt")

print("Done! Bravo!")