 
import torch
from tqdm.auto import tqdm

from modelcodes.llama3 import (
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
lr_al = 512
# PATH_S = f"./checkpoints/ablations0807/qk{rr_qk}vo{rr_vo}/"
PATH_S = "./checkpoints/240812a_dist/"

from transformers import Trainer

batch_size = 1
epochs = 1
entries = 1000000
# entries = 10
log_freq = 0.0001
sav_freq = 0.05

config = MLAConfig(reduce_ratio_qk=rr_qk, reduce_ratio_vo=rr_vo, 
                   lora_rank=lr_rk, lora_alpha=lr_al, 
                   path=PATH_S)
_, tokenizer, model, _ = get_model(config, False)

# config = MLAConfig("qkvo", 2, 128, 256, PATH_L)
# device, tokenizer, model = get_model(config, False)

from accelerate import Accelerator
accelerator = Accelerator()
device = accelerator.device

model.set_training(True)
# model = model.to(device)

dataloader, _, _ = get(batch_size)
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

optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=2e-5, eps=2e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=min(len(dataloader) * epochs, entries))

dataloader, model, optimizer, scheduler = accelerator.prepare(dataloader, model, optimizer, scheduler)

final_steps = min(len(dataloader) * epochs, entries // 8)
log_gap = int(log_freq * final_steps)
sav_gap = int(sav_freq * final_steps)

# model, optimizer, training_dataloader = accelerator.prepare(
#     model, optimizer, dataloader
# )

neg100 = (torch.ones([4096 + 1]) * -100).to(torch.int)
losses = []

for epoch in range(epochs):
    for index, data in tqdm(enumerate(dataloader), 
                            desc=f"Fitting MLA {epoch + 1} / {epochs}", 
                            total=final_steps,
                            disable = not accelerator.is_local_main_process
                            ):
        
        if index >= final_steps: break

        inputs = tokenizer(data["text"], return_tensors="pt", padding="max_length", truncation=True)
        # set_length_for_training(inputs.attention_mask.sum(dim=-1))
        lengths = inputs.attention_mask.sum(dim=-1)
        local_max_length = inputs.input_ids.shape[-1]
        labels = torch.stack([torch.cat([neg100[:local_max_length - length], input_id[-length:]])
                                for length, input_id in zip(lengths, inputs.input_ids)]).to(device)
        inputs = inputs.to(device)

        optimizer.zero_grad()
        outputs = model(**inputs, labels=labels)
        loss = outputs.loss
        # loss.backward()
        accelerator.backward(loss)

        optimizer.step()
        scheduler.step()
        # clear_total_loss()

        if index % log_gap == 0 and accelerator.is_local_main_process:
            losses.append(loss.detach().item())
            with open(PATH_S + "losses.txt", "+a") as file:
                print(loss, file=file)
        if index % sav_gap == 0:
            accelerator.wait_for_everyone()
            if accelerator.is_local_main_process:
                unwrapped_model = accelerator.unwrap_model(model)
                torch.save(unwrapped_model.state_dict(), PATH_S + 'model_state.pt')

accelerator.wait_for_everyone()
if accelerator.is_local_main_process:
    unwrapped_model = accelerator.unwrap_model(model)
    torch.save(unwrapped_model.state_dict(), PATH_S + 'model_state.pt')

    tensored_loss = torch.Tensor(losses)
    torch.save(tensored_loss, PATH_S + "losses.pt")

print("Done! Bravo!")