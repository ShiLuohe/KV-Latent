import torch
from transformers import LlamaForCausalLM, LlamaTokenizerFast

from modelcodes.llama3 import (
    model, 
    tokenizer, 
    device, 
    set_length_for_training, 
    get_total_loss, 
    clear_total_loss,
    save_model,
    load_model
)

PROMPT = "Hello world! This is llama-3-8B by META."

inputs = tokenizer(PROMPT, return_tensors="pt", padding="max_length").to(device)
set_length_for_training(inputs.attention_mask.sum(dim=-1))

outputs = model(**inputs, output_hidden_states=True)

response = tokenizer.decode(outputs.logits.argmax(dim=-1)[0])
print(response)

