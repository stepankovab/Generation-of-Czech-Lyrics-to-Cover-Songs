import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, StoppingCriteria
from dataset_types import DatasetType
from english_structure_extractor import SectionStructure
from CODE.LLMExperiments.evaluator import Evaluator
from eval.syllabator import syllabify
from HT_loader import HT_loader
import argparse
import os
from generate_whole import generate_whole
from generate_lines import generate_lines


parser = argparse.ArgumentParser()

parser.add_argument("--batch_size", default=32, type=int, help="Batch size")
parser.add_argument("--model", default="GPT2_oscar", type=str, help="tinyLlama # GPT2_oscar # GPT2_czech_XL # Mistral_czech # ") 
parser.add_argument("--epoch", default=3, type=int, help="Epoch of the trained model")
parser.add_argument("--dataset_path", default="./", type=str, help="./ # DATA/Velky_zpevnik # ")
parser.add_argument("--input_section", default="We don't talk about Bruno,No,No,No", type=str, help="Input section in English")
parser.add_argument("--from_dict", default=False, type=bool, help="Take testing data from HT dict")
parser.add_argument("--dataset_type", default=6, type=int, help="Dataset type: BASELINE = 1, SYLLABLES = 2, END_OF_LINES = 3, CHARACTERISTIC_WORDS = 4, UNRHYMED = 5, SYLLABLES_AND_WORDS = 6, SYLLABLES_AND_ENDS = 7, ENDS_AND_WORDS = 8")
parser.add_argument("--generation_method", default="whole", type=int, help="whole, lines")


args = parser.parse_args([] if "__file__" not in globals() else None)

BATCH_SIZE = args.batch_size
MODEL_NAME = args.model
EPOCH = args.epoch
DATASET_PATH = args.dataset_path
DATASET_TYPE = DatasetType(args.dataset_type)

MODEL_PATH = os.path.join(DATASET_PATH, "trained_models", f"{MODEL_NAME}_{DATASET_TYPE.name}_lyricist_{EPOCH}_{BATCH_SIZE}.pt")

print(f"----------------- {MODEL_NAME} ------ {DATASET_TYPE} ----------------")

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

print("="*10 + "  " + MODEL_PATH + " " + "="*10)

if args.from_dict == True:
    input_sections = HT_loader(args.dataset_path)
else:
    input_sections = [args.input_section]

if args.generation_method == "whole":
    generate_whole(model, MODEL_PATH, tokenizer, input_sections, DATASET_TYPE, device)
elif args.generation_method == "lines":
    generate_lines()