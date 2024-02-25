import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from dataset_types import DatasetType
import argparse
import os


parser = argparse.ArgumentParser()
# These arguments will be set appropriately by ReCodEx, even if you change them.
parser.add_argument("--batch_size", default=32, type=int, help="Batch size")
parser.add_argument("--model", default="GPT2_oscar", type=str, help="tinyLlama # GPT2_oscar # GPT2_czech_XL # Mistral_czech # ") 
parser.add_argument("--epochs", default=5, type=int, help="Number of epochs")
parser.add_argument("--starting_epoch", default=0, type=int, help="epoch to load, 0 for not loading")
parser.add_argument("--learning_rate", default=3e-5, type=float, help="Learning rate")
parser.add_argument("--warmup_steps", default=500, type=int, help="Warmup steps")
parser.add_argument("--dataset_path", default="./", type=str, help="./ # DATA/Velky_zpevnik # ")
parser.add_argument("--dataset_type", default=1, type=int, help="Dataset type: BASELINE = 1, SYLLABLES = 2, END_OF_LINES = 3, CHARACTERISTIC_WORDS = 4, UNRHYMED = 5, SYLLABLES_AND_WORDS = 6, SYLLABLES_AND_ENDS = 7, ENDS_AND_WORDS = 8")


args = parser.parse_args([] if "__file__" not in globals() else None)

BATCH_SIZE = args.batch_size
MODEL_NAME = args.model
EPOCHS = args.epochs
DATASET_PATH = args.dataset_path
DATASET_TYPE = DatasetType(args.dataset_type)

print(f"----------------- {MODEL_NAME} ------ {DATASET_TYPE} ----------------")


model_paths = [f"{MODEL_NAME}_{DATASET_TYPE.name}_lyricist_{i}_{BATCH_SIZE}.pt" for i in range(args.starting_epoch, EPOCHS)]

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

    model.load_state_dict(state_dict=torch.load(os.path.join(DATASET_PATH, "trained_models", model_path), map_location=torch.device(device)))
    model.eval()

    
    if DATASET_TYPE == DatasetType.BASELINE:
        text = ""

    elif DATASET_TYPE == DatasetType.SYLLABLES:
        text = "4 6 4 6 #\n"
                
    elif DATASET_TYPE == DatasetType.END_OF_LINES:
        text = "ví nám ní ky #\n"
            
    elif DATASET_TYPE == DatasetType.CHARACTERISTIC_WORDS:
        text = "4 # čas půlnoc komáři kostel kytka #\n"

    elif DATASET_TYPE == DatasetType.UNRHYMED:
        pass

    elif DATASET_TYPE == DatasetType.SYLLABLES_AND_WORDS:
        text = "4 6 4 8 # čas půlnoc komáři kostel kytka #\n"

    elif DATASET_TYPE == DatasetType.SYLLABLES_AND_ENDS:
        text = "4 6 4 6 # ví nám ní ky #\n"

    elif DATASET_TYPE == DatasetType.ENDS_AND_WORDS:
        text = "ví nám tel sám # čas půlnoc komáři kostel kytka #\n"

    elif DATASET_TYPE == DatasetType.FORCED_SYLLABLES:
        text = "4 6 4 8 #\n"

    else:
        raise Exception(f"We don't support a Dataset type {DATASET_TYPE}")

    inputs = tokenizer(text, return_tensors="pt") 
    
    tokenizer.encode(text, return_tensors="pt") #directly for input_ids

    # model output using Top-k sampling text generation method
    sample_outputs = model.generate(inputs.input_ids,
                                    pad_token_id=50256,
                                    do_sample=True, 
                                    max_length=100, # put the token number you want
                                    top_k=40,
                                    num_return_sequences=15)

    # generated sequence
    for i, sample_output in enumerate(sample_outputs):
        print("\n{}\n\n{}".format(i+1, tokenizer.decode(sample_output.tolist()))) # tokenizer.decode(sample_output, skip_special_tokens=True)

