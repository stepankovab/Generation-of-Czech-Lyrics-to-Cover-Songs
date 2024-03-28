import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, StoppingCriteria
from dataset_types import DatasetType
from english_structure_extractor import SectionStructure
from postprocessing import Postprocesser
from eval.syllabator import syllabify
import os
import re
import requests


class StoppingSequenceCriteria(StoppingCriteria):
    def __init__(self, prompt, tokenizer):
        self.prompt=prompt
        self.tokenizer = tokenizer

    def __call__(self, input_ids, scores, **kwargs):
        generated_text = self.tokenizer.decode(input_ids[0])
        generated_text = generated_text.replace(self.prompt,'')
        generated_lines = generated_text.strip().split("\n")
        if (len(generated_lines) > len(self.prompt.split("\n"))):
            return True  # Stop generation
        return False  # Continue generation

    def __len__(self):
        return 1

    def __iter__(self):
        yield self

def prepare_prompt(dataset_type, structure: SectionStructure, line_i, ending = None):
    if dataset_type == DatasetType.CHARACTERISTIC_WORDS:
        prompt = f"{structure.line_keywords[line_i]} # "
    elif dataset_type == DatasetType.ENDS_AND_WORDS:
        prompt = f"{structure.line_keywords[line_i]}\n{ending} # "
    elif dataset_type == DatasetType.BASELINE:
        prompt = ""
    elif dataset_type == DatasetType.END_OF_LINES:
        prompt = f"{ending} # "
    elif dataset_type == DatasetType.SYLLABLES:
        prompt = f"{structure.syllables[line_i]} # "
    elif dataset_type == DatasetType.SYLLABLES_AND_ENDS:
        prompt = f"{structure.syllables[line_i]} # {ending} # "
    elif dataset_type == DatasetType.SYLLABLES_AND_WORDS:
        prompt = f"{structure.line_keywords[line_i]}\n{structure.syllables[line_i]} # "
    elif dataset_type == DatasetType.SYLLABLES_ENDS_WORDS:
        prompt = f"{structure.line_keywords[line_i]}\n{structure.syllables[line_i]} # {ending} # "
    elif dataset_type == DatasetType.UNRHYMED_LEN:
        url = 'http://lindat.mff.cuni.cz/services/translation/api/v2/models/en-cs'
        response = requests.post(url, data = {"input_text": structure.original_lyrics[line_i]})
        response.encoding='utf8'
        translated_output = ''.join([x for x in response.text if x.isalpha() or x.isspace()]).strip()
        prompt = f"{len(syllabify(translated_output))} # {translated_output}\n{structure.syllables[line_i]} # "
    elif dataset_type == DatasetType.UNRHYMED_LEN_END:
        url = 'http://lindat.mff.cuni.cz/services/translation/api/v2/models/en-cs'
        response = requests.post(url, data = {"input_text": structure.original_lyrics[line_i]})
        response.encoding='utf8'
        translated_output = ''.join([x for x in response.text if x.isalpha() or x.isspace()]).strip()
        translated_output_sylls = syllabify(translated_output)
        prompt = f"{len(translated_output_sylls)} # {translated_output_sylls[-1][-min(len(translated_output_sylls[-1]), 3):]} # {translated_output}\n{structure.syllables[line_i]} # {ending} # "
    else:
        raise Exception(f"We don't support a Dataset type {dataset_type}")

    return prompt


def extract_model_out(model_out, prompt):
    if len(model_out) <= len(prompt):
        return ""
    
    assert model_out[:len(prompt)] == prompt
    model_out = model_out[len(prompt):]
    out_lines = model_out.strip().split("\n")
    return re.sub(',', '', out_lines[0])


def generate_lines(args, input_sections):
    wout_dataset_type = DatasetType(args.dataset_type)

    if wout_dataset_type == DatasetType.CHARACTERISTIC_WORDS:
        w_dataset_type = DatasetType.ENDS_AND_WORDS
    elif wout_dataset_type == DatasetType.BASELINE:
        w_dataset_type = DatasetType.END_OF_LINES
    elif wout_dataset_type == DatasetType.SYLLABLES:
        w_dataset_type = DatasetType.SYLLABLES_AND_ENDS
    elif wout_dataset_type == DatasetType.SYLLABLES_AND_WORDS:
        w_dataset_type = DatasetType.SYLLABLES_ENDS_WORDS
    elif wout_dataset_type == DatasetType.UNRHYMED_LEN:
        w_dataset_type = DatasetType.UNRHYMED_LEN_END
    else:
        raise Exception(f"We don't support a Dataset type {wout_dataset_type}")

    device = 'cpu'
    if torch.cuda.is_available():
        print("cuda available.")
        device = 'cuda'

    if args.model == "GPT2_oscar":
        tokenizer = AutoTokenizer.from_pretrained("lchaloupsky/czech-gpt2-oscar")
        wout_model = AutoModelForCausalLM.from_pretrained("lchaloupsky/czech-gpt2-oscar")
        w_model = AutoModelForCausalLM.from_pretrained("lchaloupsky/czech-gpt2-oscar")
        tokenizer.model_max_length=1024

    elif args.model == "GPT2_czech_XL":
        tokenizer = AutoTokenizer.from_pretrained("BUT-FIT/Czech-GPT-2-XL-133k")
        wout_model = AutoModelForCausalLM.from_pretrained("BUT-FIT/Czech-GPT-2-XL-133k")
        w_model = AutoModelForCausalLM.from_pretrained("BUT-FIT/Czech-GPT-2-XL-133k")
        tokenizer.model_max_length=1024

    elif args.model == "Mistral_czech":
        tokenizer = AutoTokenizer.from_pretrained("simecek/cswikimistral_0.1")
        wout_model = AutoModelForCausalLM.from_pretrained("simecek/cswikimistral_0.1")
        w_model = AutoModelForCausalLM.from_pretrained("simecek/cswikimistral_0.1")
        tokenizer.model_max_length=1024

    elif args.model == "tinyLlama":
        tokenizer = AutoTokenizer.from_pretrained("TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T")
        wout_model = AutoModelForCausalLM.from_pretrained("TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T")
        w_model = AutoModelForCausalLM.from_pretrained("TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T")
        tokenizer.model_max_length=1024

    wout_model_path = os.path.join(args.model_path, f"{args.model}_{wout_dataset_type.name}_{args.generation_method}_{args.epoch}.pt")
    w_model_path = os.path.join(args.model_path, f"{args.model}_{w_dataset_type.name}_{args.generation_method}_{args.epoch}.pt")
    
    print("="*10 + "  " + wout_model_path + " " + "="*10)
    wout_model.load_state_dict(state_dict=torch.load(wout_model_path, map_location=torch.device(device)))
    print("="*10 + "  " + w_model_path + " " + "="*10)
    w_model.load_state_dict(state_dict=torch.load(w_model_path, map_location=torch.device(device)))
    wout_model.eval()
    w_model.eval()

    structure = SectionStructure()
    postprocesser = Postprocesser()

    result_pairs = []

    for input_section in input_sections:
        # Load the structure of the english text
        structure.fill(input_section) 

        result = []

        known_endings = {}

        for line_i in range(structure.num_lines):

            if structure.rhyme_scheme[line_i] in known_endings:
                prompt = prepare_prompt(w_dataset_type, structure, line_i, known_endings[structure.rhyme_scheme[line_i]])

                print(prompt)
                inputs = tokenizer(prompt, return_tensors="pt") 
                tokenizer.encode(prompt, return_tensors="pt")
        
                # model output using Top-k sampling text generation method
                sample_outputs = w_model.generate(**inputs,
                    do_sample=True,
                    pad_token_id=tokenizer.eos_token_id,
                    num_return_sequences=args.out_per_gerenation,
                    penalty_alpha=0.6,
                    max_new_tokens=128,
                    stopping_criteria=StoppingSequenceCriteria(prompt, tokenizer),
                    )
                
                out_lines = [extract_model_out(tokenizer.decode(sample_output.tolist(), skip_special_tokens=True), prompt) for sample_output in sample_outputs]
                model_out = postprocesser.choose_best_line(out_lines, syllables_in=structure.syllables[line_i], ending_in=known_endings[structure.rhyme_scheme[line_i]], keywords_in=structure.en_line_keywords[line_i], keywords_in_en=True, text_in=structure.original_lyrics[line_i], text_in_english=True)
                
                print(f"\n{model_out}\n")
                result.append(model_out)

            else:
                prompt = prepare_prompt(wout_dataset_type, structure, line_i)

                print(prompt)
                inputs = tokenizer(prompt, return_tensors="pt") 
                tokenizer.encode(prompt, return_tensors="pt")
        
                # model output using Top-k sampling text generation method
                sample_outputs = wout_model.generate(**inputs,
                    do_sample=True,
                    pad_token_id=tokenizer.eos_token_id,
                    num_return_sequences=args.out_per_gerenation,
                    penalty_alpha=0.6,
                    max_new_tokens=128,
                    stopping_criteria=StoppingSequenceCriteria(prompt, tokenizer),
                    )
                
                out_lines = [extract_model_out(tokenizer.decode(sample_output.tolist(), skip_special_tokens=True), prompt) for sample_output in sample_outputs]
                model_out = postprocesser.choose_best_line(out_lines, syllables_in=structure.syllables[line_i], keywords_in=structure.en_line_keywords[line_i], keywords_in_en=True, text_in=structure.original_lyrics[line_i], text_in_english=True)
                
                print(f"\n{model_out}\n")
                result.append(model_out)

                syll_output = syllabify(model_out)
                known_endings[structure.rhyme_scheme[line_i]] = syll_output[-1][-min(len(syll_output[-1]), 3):]

        result_pairs.append((','.join(result), structure.copy()))
        for line in result:
            print(line)
        print()

    return result_pairs


