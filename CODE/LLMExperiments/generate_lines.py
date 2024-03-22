import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, StoppingCriteria
from dataset_types import DatasetType
from english_structure_extractor import SectionStructure
from eval.syllabator import syllabify
import os
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
        prompt = f"{structure.line_keywords[line_i]} #\n{ending} # "
    elif dataset_type == DatasetType.BASELINE:
        prompt = ""
    elif dataset_type == DatasetType.END_OF_LINES:
        prompt = f"{ending} # "
    elif dataset_type == DatasetType.SYLLABLES:
        prompt = f"{structure.syllables[line_i]} # "
    elif dataset_type == DatasetType.SYLLABLES_AND_ENDS:
        prompt = f"{structure.syllables[line_i]} # {ending} # "
    elif dataset_type == DatasetType.SYLLABLES_AND_WORDS:
        prompt = f"{structure.line_keywords[line_i]} #\n{structure.syllables[line_i]} # "
    elif dataset_type == DatasetType.SYLLABLES_ENDS_WORDS:
        prompt = f"{structure.line_keywords[line_i]} #\n{structure.syllables[line_i]} # {ending} # "
    elif dataset_type == DatasetType.UNRHYMED_LEN:
        url = 'http://lindat.mff.cuni.cz/services/translation/api/v2/models/en-cs'
        response = requests.post(url, data = {"input_text": structure.original_lyrics[line_i]})
        response.encoding='utf8'
        translated_output = ''.join([x for x in response.text if x.isalpha() or x.isspace()]).strip()
        prompt = f"{len(syllabify(translated_output))} # {translated_output} #\n{structure.syllables[line_i]} # "
    elif dataset_type == DatasetType.UNRHYMED_LEN_END:
        url = 'http://lindat.mff.cuni.cz/services/translation/api/v2/models/en-cs'
        response = requests.post(url, data = {"input_text": structure.original_lyrics[line_i]})
        response.encoding='utf8'
        translated_output = ''.join([x for x in response.text if x.isalpha() or x.isspace()]).strip()
        translated_output_sylls = syllabify(translated_output)
        prompt = f"{len(translated_output_sylls)} # {translated_output_sylls[-1][-min(len(translated_output_sylls[-1]), 3):]} # {translated_output} #\n{structure.syllables[line_i]} # {ending} # "
    else:
        prompt = ""

    return prompt


def generate_lines(args, input_sections):

    LINES_PER_GENERATION = 5

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

    wout_model_path = os.path.join(args.dataset_path, "trained_models", f"{args.model}_{wout_dataset_type.name}_one_line_lyricist_{args.epoch}.pt")
    w_model_path = os.path.join(args.dataset_path, "trained_models", f"{args.model}_{w_dataset_type.name}_one_line_lyricist_{args.epoch}.pt")
    
    print("="*10 + "  " + wout_model_path + " " + "="*10)
    wout_model.load_state_dict(state_dict=torch.load(wout_model_path, map_location=torch.device(device)))
    w_model.load_state_dict(state_dict=torch.load(w_model_path, map_location=torch.device(device)))
    wout_model.eval()
    w_model.eval()

    structure = SectionStructure()

    result_pairs = []

    for input_section in input_sections:
        # Load the structure of the english text
        structure.fill(input_section) 

        temp_result = [[] for x in range(LINES_PER_GENERATION)]

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
                    num_return_sequences=LINES_PER_GENERATION,
                    penalty_alpha=0.6,
                    max_new_tokens=128,
                    stopping_criteria=StoppingSequenceCriteria(prompt, tokenizer),
                    )

                # generated sequence
                for i, sample_output in enumerate(sample_outputs):
                    model_out = tokenizer.decode(sample_output.tolist(), skip_special_tokens=True)
                    print("\n{}\n\n{}\n".format(i+1, model_out)) # tokenizer.decode(sample_output, skip_special_tokens=True)
                    temp_result[i].append(''.join([x for x in model_out.strip().split("\n")[len(prompt.split("\n")) - 1] if x.isalpha() or x.isspace() or x == '.' or x == ',']).strip())
                    
            else:
                prompt = prepare_prompt(wout_dataset_type, structure, line_i)

                print(prompt)
                inputs = tokenizer(prompt, return_tensors="pt") 
                tokenizer.encode(prompt, return_tensors="pt")
        
                # model output using Top-k sampling text generation method
                sample_outputs = wout_model.generate(**inputs,
                    do_sample=True,
                    pad_token_id=tokenizer.eos_token_id,
                    num_return_sequences=LINES_PER_GENERATION,
                    penalty_alpha=0.6,
                    max_new_tokens=128,
                    stopping_criteria=StoppingSequenceCriteria(prompt, tokenizer),
                    )

                # generated sequence
                for i, sample_output in enumerate(sample_outputs):
                    model_out = tokenizer.decode(sample_output.tolist(), skip_special_tokens=True)
                    print("\n{}\n\n{}\n".format(i+1, model_out)) # tokenizer.decode(sample_output, skip_special_tokens=True)

                    temp_result[i].append(''.join([x for x in model_out.strip().split("\n")[len(prompt.split("\n")) - 1] if x.isalpha() or x.isspace()]).strip())
                    
                    if i == 0 and temp_result[i][line_i] != "":
                        syll_output = syllabify(temp_result[i][line_i])
                        known_endings[structure.rhyme_scheme[line_i]] = syll_output[-1][-min(len(syll_output[-1]), 3):]

        for result in temp_result:
            result_pairs.append((','.join(result), structure.copy()))
            for line in result:
                print(line)
            print()

    return result_pairs


