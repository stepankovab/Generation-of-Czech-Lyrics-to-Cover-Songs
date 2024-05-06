import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, StoppingCriteria
from dataset_types import DatasetType
from english_structure_extractor import SectionStructureExtractor, SectionStructure
from postprocessing import Postprocesser
from evaluator import Evaluator
from rhymer_types import RhymerType
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
    if dataset_type == DatasetType.BASELINE:
        prompt = " "
    elif dataset_type == DatasetType.SYLLABLES:
        prompt = f"{structure.syllables[line_i]} # "
    elif dataset_type == DatasetType.SYLLABLES_ENDS:
        prompt = f"{structure.syllables[line_i]} # {ending} # "
    elif dataset_type == DatasetType.WORDS:
        prompt = f"{structure.line_keywords[line_i]} # "
    elif dataset_type == DatasetType.WORDS_ENDS:
        prompt = f"{structure.line_keywords[line_i]}\n{ending} # "
    elif dataset_type == DatasetType.SYLLABLES_WORDS:
        prompt = f"{structure.line_keywords[line_i]}\n{structure.syllables[line_i]} # "
    elif dataset_type == DatasetType.SYLLABLES_WORDS_ENDS:
        prompt = f"{structure.line_keywords[line_i]}\n{structure.syllables[line_i]} # {ending} # "
    elif dataset_type == DatasetType.UNRHYMED_LEN:
        url = 'http://lindat.mff.cuni.cz/services/translation/api/v2/models/en-cs'
        response = requests.post(url, data = {"input_text": structure.original_lyrics_list[line_i]})
        response.encoding='utf8'
        translated_output = ''.join([x for x in response.text if x.isalpha() or x.isspace()]).strip()
        prompt = f"{len(syllabify(translated_output))} # {translated_output}\n{structure.syllables[line_i]} # "
    elif dataset_type == DatasetType.UNRHYMED_LEN_END:
        url = 'http://lindat.mff.cuni.cz/services/translation/api/v2/models/en-cs'
        response = requests.post(url, data = {"input_text": structure.original_lyrics_list[line_i]})
        response.encoding='utf8'
        translated_output = ''.join([x for x in response.text if x.isalpha() or x.isspace()]).strip()
        translated_output_sylls = syllabify(translated_output)
        prompt = f"{len(translated_output_sylls)} # {translated_output_sylls[-1][-min(len(translated_output_sylls[-1]), 3):]} # {translated_output}\n{structure.syllables[line_i]} # {ending} # "
    elif dataset_type == DatasetType.FORCED_SYLLABLES:
        prompt = f"{structure.syllables[line_i]} # "
    elif dataset_type == DatasetType.FORCED_SYLLABLES_ENDS:
        prompt = f"{structure.syllables[line_i]} # {ending} # "
    else:
        raise Exception(f"We don't support a Dataset type {dataset_type}")

    return prompt


def extract_model_out(model_out, prompt):
    if len(model_out) <= len(prompt):
        return ""
    
    model_out = model_out[len(prompt):]
    out_lines = model_out.strip().split("\n")
    return re.sub(',', '', out_lines[0])


def generate_lines(args, input_sections, verbose=False):
    wout_dataset_type = DatasetType(args.dataset_type)

    
    if wout_dataset_type == DatasetType.BASELINE:
        w_dataset_type = DatasetType.BASELINE
    elif wout_dataset_type == DatasetType.WORDS:
        w_dataset_type = DatasetType.WORDS_ENDS
    elif wout_dataset_type == DatasetType.SYLLABLES:
        w_dataset_type = DatasetType.SYLLABLES_ENDS
    elif wout_dataset_type == DatasetType.SYLLABLES_WORDS:
        w_dataset_type = DatasetType.SYLLABLES_WORDS_ENDS
    elif wout_dataset_type == DatasetType.UNRHYMED_LEN:
        w_dataset_type = DatasetType.UNRHYMED_LEN_END
    elif wout_dataset_type == DatasetType.FORCED_SYLLABLES:
        w_dataset_type = DatasetType.FORCED_SYLLABLES_ENDS
    else:
        raise Exception(f"We don't support a Dataset type {wout_dataset_type}")

    device = 'cpu'
    if torch.cuda.is_available():
        if verbose:
            print("cuda available.")
        device = 'cuda'

    if args.model == "OSCAR_GPT2":
        tokenizer = AutoTokenizer.from_pretrained("lchaloupsky/czech-gpt2-oscar")
        wout_model = AutoModelForCausalLM.from_pretrained("lchaloupsky/czech-gpt2-oscar")
        w_model = AutoModelForCausalLM.from_pretrained("lchaloupsky/czech-gpt2-oscar")

    elif args.model == "VUT_GPT2":
        tokenizer = AutoTokenizer.from_pretrained("BUT-FIT/Czech-GPT-2-XL-133k")
        wout_model = AutoModelForCausalLM.from_pretrained("BUT-FIT/Czech-GPT-2-XL-133k")
        w_model = AutoModelForCausalLM.from_pretrained("BUT-FIT/Czech-GPT-2-XL-133k")

    elif args.model == "TINYLLAMA":
        tokenizer = AutoTokenizer.from_pretrained("TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T")
        wout_model = AutoModelForCausalLM.from_pretrained("TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T")
        w_model = AutoModelForCausalLM.from_pretrained("TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T")

    elif args.model == "VUT_TINYLLAMA":
        tokenizer = AutoTokenizer.from_pretrained("BUT-FIT/CSTinyLlama-1.2B")
        wout_model = AutoModelForCausalLM.from_pretrained("BUT-FIT/CSTinyLlama-1.2B")
        w_model = AutoModelForCausalLM.from_pretrained("BUT-FIT/CSTinyLlama-1.2B")
            
    else:
        raise ValueError(f"Model {args.model} is not supported with generation mode 'lines'.")

    # Set special tokens if they are not already set
    if tokenizer.sep_token is None:
        tokenizer.add_special_tokens({'sep_token': '[SEP]'})
    if tokenizer.cls_token is None:
        tokenizer.add_special_tokens({'cls_token': '[CLS]'})
    if tokenizer.mask_token is None:
        tokenizer.add_special_tokens({'mask_token': '[MASK]'})

    wout_model_path = os.path.join(args.model_path, f"{args.model}_{wout_dataset_type.name}_{args.generation_method}_{args.epoch}.pt")
    w_model_path = os.path.join(args.model_path, f"{args.model}_{w_dataset_type.name}_{args.generation_method}_{args.epoch}.pt")
    
    if verbose:
        print("="*10 + "  " + wout_model_path + " " + "="*10)
    wout_model.load_state_dict(state_dict=torch.load(wout_model_path, map_location=torch.device(device)))
    wout_model.to(device)
    if verbose:
        print("="*10 + "  " + w_model_path + " " + "="*10)
    w_model.load_state_dict(state_dict=torch.load(w_model_path, map_location=torch.device(device)))
    w_model.to(device)
    wout_model.eval()
    w_model.eval()

    structure_extractor = SectionStructureExtractor(english_rhyme_detector=RhymerType(args.rhymer))
    postprocesser = Postprocesser(evaluator=Evaluator(czech_rhyme_detector=RhymerType(args.rhymer)))

    result_pairs = []

    for in_sec_id in range(len(input_sections)):
        input_section = input_sections[in_sec_id]
        
        if isinstance(input_section, SectionStructure):
            structure = input_section
        else:
            structure = structure_extractor.create_section_structure(input_section) 
            
        result = []
        known_endings = {}

        for line_i in range(structure.num_lines):

            if structure.rhyme_scheme[line_i] in known_endings:
                prompt = prepare_prompt(w_dataset_type, structure, line_i, known_endings[structure.rhyme_scheme[line_i]])
                
                inputs = tokenizer([prompt],return_token_type_ids=False, return_tensors="pt").to(device)

                sample_outputs = w_model.generate(**inputs,
                    do_sample=True,
                    top_p=0.95,
                    repetition_penalty=1.0,
                    temperature=0.8,
                    max_new_tokens=256,
                    num_return_sequences=args.out_per_generation,
                    pad_token_id=tokenizer.eos_token_id,
                    penalty_alpha=0.6,
                    stopping_criteria=StoppingSequenceCriteria(prompt, tokenizer),
                    )
                
                out_lines = [extract_model_out(tokenizer.decode(sample_output.tolist(), skip_special_tokens=True), prompt) for sample_output in sample_outputs]
                model_out = postprocesser.choose_best_line(out_lines, syllables_in=structure.syllables[line_i], ending_in=known_endings[structure.rhyme_scheme[line_i]], text_in=structure.original_lyrics_list[line_i], text_in_english=True, remove_add_stopwords=args.postprocess_stopwords)
                
                result.append(model_out)

            else:
                prompt = prepare_prompt(wout_dataset_type, structure, line_i)

                inputs = tokenizer([prompt],return_token_type_ids=False, return_tensors="pt").to(device)
        
                # model output using Top-k sampling text generation method
                sample_outputs = wout_model.generate(**inputs,
                    do_sample=True,
                    top_p=0.95,
                    repetition_penalty=1.0,
                    temperature=0.8,
                    max_new_tokens=256,
                    num_return_sequences=args.out_per_generation,
                    pad_token_id=tokenizer.eos_token_id,
                    penalty_alpha=0.6,
                    stopping_criteria=StoppingSequenceCriteria(prompt, tokenizer),
                    )
                
                out_lines = [extract_model_out(tokenizer.decode(sample_output.tolist(), skip_special_tokens=True), prompt) for sample_output in sample_outputs]
                model_out = postprocesser.choose_best_line(out_lines, syllables_in=structure.syllables[line_i], text_in=structure.original_lyrics_list[line_i], text_in_english=True, remove_add_stopwords=args.postprocess_stopwords)
                
                result.append(model_out)

                syll_output = syllabify(model_out)
                if len(syll_output) > 0:
                    known_endings[structure.rhyme_scheme[line_i]] = syll_output[-1][-min(len(syll_output[-1]), 3):]

        result_pairs.append((','.join(result), structure))

        if verbose:
            print("lyrics: \n\n")
            for line in result:
                print(line)
            print()

    return result_pairs


