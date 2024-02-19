import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from enum import Enum

class DatasetType(Enum):
    BASELINE = 1
    SYLLABLES = 2
    END_OF_LINES = 3
    CHARACTERISTIC_WORD = 4


MODEL_NAME = "tinyLlama" # "GPT2_oscar" # "GPT2_czech_XL" # "Mistral_czech" # 
DATASET_TYPE = DatasetType.END_OF_LINES

model_paths = [f'./{MODEL_NAME}_{DATASET_TYPE.name}_lyricist_{i}.pt' for i in range(2,6)]

device = 'cpu'
if torch.cuda.is_available():
    print("cuda available.")
    device = 'cuda'

if MODEL_NAME == "GPT2_oscar":
    tokenizer = AutoTokenizer.from_pretrained("lchaloupsky/czech-gpt2-oscar")
    model = AutoModelForCausalLM.from_pretrained("lchaloupsky/czech-gpt2-oscar")
    tokenizer.model_max_length=1024

elif MODEL_NAME == "GPT2_czech_XL":
    tokenizer = AutoTokenizer.from_pretrained("BUT-FIT/Czech-GPT-2-XL-133k")
    model = AutoModelForCausalLM.from_pretrained("BUT-FIT/Czech-GPT-2-XL-133k")
    tokenizer.model_max_length=1024

elif MODEL_NAME == "Mistral_czech":
    tokenizer = AutoTokenizer.from_pretrained("simecek/cswikimistral_0.1")
    model = AutoModelForCausalLM.from_pretrained("simecek/cswikimistral_0.1")
    tokenizer.model_max_length=1024

elif MODEL_NAME == "tinyLlama":
    tokenizer = AutoTokenizer.from_pretrained("TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T")
    model = AutoModelForCausalLM.from_pretrained("TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T")
    tokenizer.model_max_length=1024

for model_path in model_paths:

    print("="*10 + "  " + model_path + " " + "="*10)

    model.load_state_dict(state_dict=torch.load(model_path, map_location=torch.device(device)))
    model.eval()

    # input sequence
    text = "ky ní ky ní"
    inputs = tokenizer(text, return_tensors="pt") # tokenizer.encode(text, return_tensors="pt") directly for input_ids

    # model output using Top-k sampling text generation method
    sample_outputs = model.generate(inputs.input_ids,
                                    pad_token_id=50256,
                                    do_sample=True, 
                                    max_length=50, # put the token number you want
                                    top_k=40,
                                    num_return_sequences=6)

    # generated sequence
    for i, sample_output in enumerate(sample_outputs):
        print("{}\n\n{}".format(i+1, tokenizer.decode(sample_output.tolist()))) # tokenizer.decode(sample_output, skip_special_tokens=True)

