import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
    
device = 'cpu'
if torch.cuda.is_available():
    print("cuda available.")
    device = 'cuda'

def generate_basic_output(model_name, prompt):

    model, tokenizer = AutoModelForCausalLM.from_pretrained(model_name), AutoTokenizer.from_pretrained(model_name)
    model.to(device)
    print(f"loaded model: {model_name}")

    # Set special tokens if they are not already set
    if tokenizer.sep_token is None:
        tokenizer.add_special_tokens({'sep_token': '[SEP]'})
    if tokenizer.cls_token is None:
        tokenizer.add_special_tokens({'cls_token': '[CLS]'})
    if tokenizer.mask_token is None:
        tokenizer.add_special_tokens({'mask_token': '[MASK]'})

    inputs = tokenizer([prompt],return_token_type_ids=False, return_tensors="pt").to(device)

    output_ids = model.generate(**inputs,
        do_sample=True,
        top_p=0.95,
        repetition_penalty=1.0,
        temperature=0.8,
        max_new_tokens=128,
        num_return_sequences=1,
        pad_token_id=tokenizer.eos_token_id,
        penalty_alpha=0.6
        )
    
    for o_ids in output_ids:
        print(tokenizer.decode(o_ids.tolist(), skip_special_tokens=True))

    print()


models = ["lchaloupsky/czech-gpt2-oscar"] # ['../models/csmpt7b', '../models/cswikimistral_0.1']

prompt = "Zapadající slunce"
for model in models:
    generate_basic_output(model, prompt)
