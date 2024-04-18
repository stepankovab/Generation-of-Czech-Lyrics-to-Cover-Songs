import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, StoppingCriteria
from dataset_types import DatasetType
from rhymer_types import RhymerType
from english_structure_extractor import SectionStructure
from postprocessing import Postprocesser
from evaluator import Evaluator
from lyrics_datasets import prepare_prompt_whole, extract_output_whole
import os
import re
import json


class StoppingSequenceCriteria(StoppingCriteria):
    def __init__(self, prompt, tokenizer, desired_lines):
        self.prompt=prompt
        self.tokenizer = tokenizer
        self.desired_lines = desired_lines

    def __call__(self, input_ids, scores, **kwargs):
        generated_text = self.tokenizer.decode(input_ids[0])
        generated_text = generated_text.replace(self.prompt,'')
        generated_lines = generated_text.strip().split("\n")
        if len(generated_lines) > self.desired_lines:
            return True  # Stop generation
        return False  # Continue generation

    def __len__(self):
        return 1

    def __iter__(self):
        yield self


def generate_whole(args, input_sections):
    """
    input section can be either string or sectionStucture
    """
    
    dataset_type = DatasetType(args.dataset_type)
    
    device = 'cpu'
    if torch.cuda.is_available():
        print("cuda available.")
        device = 'cuda'

    if args.model == "OSCAR_GPT2":
        tokenizer = AutoTokenizer.from_pretrained("lchaloupsky/czech-gpt2-oscar")
        model = AutoModelForCausalLM.from_pretrained("lchaloupsky/czech-gpt2-oscar")

    elif args.model == "VUT_GPT2":
        tokenizer = AutoTokenizer.from_pretrained("BUT-FIT/Czech-GPT-2-XL-133k")
        model = AutoModelForCausalLM.from_pretrained("BUT-FIT/Czech-GPT-2-XL-133k")

    elif args.model == "TINYLLAMA":
        tokenizer = AutoTokenizer.from_pretrained("TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T")
        model = AutoModelForCausalLM.from_pretrained("TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T")
    
    elif args.model == "VUT_TINYLLAMA":
        tokenizer = AutoTokenizer.from_pretrained("BUT-FIT/CSTinyLlama-1.2B")
        model = AutoModelForCausalLM.from_pretrained("BUT-FIT/CSTinyLlama-1.2B")

    else:
        raise ValueError(f"Model {args.model} is not supported with generation mode 'whole'.")

    # Set special tokens if they are not already set
    if tokenizer.sep_token is None:
        tokenizer.add_special_tokens({'sep_token': '[SEP]'})
    if tokenizer.cls_token is None:
        tokenizer.add_special_tokens({'cls_token': '[CLS]'})
    if tokenizer.mask_token is None:
        tokenizer.add_special_tokens({'mask_token': '[MASK]'})

    model_path = os.path.join(args.model_path, f"{args.model}_{dataset_type.name}_{args.generation_method}_{args.epoch}.pt")

    print("="*10 + "  " + model_path + " " + "="*10)
    model.load_state_dict(state_dict=torch.load(model_path, map_location=torch.device(device)))
    model.to(device)
    model.eval()

    structure = SectionStructure(english_rhyme_detector=RhymerType(args.rhymer))
    postprocesser = Postprocesser(evaluator=Evaluator(czech_rhyme_detector=RhymerType(args.rhymer)))

    result_pairs = []

    if args.outsource_rhyme_schemes and args.from_dict:
        with open("english_HT_rhymes_espeak.json", "r", encoding="utf-8") as json_file:
            espeak_rhymes = json.load(json_file)

        assert len(espeak_rhymes) == len(input_sections)

    for in_sec_id in range(len(input_sections)):
        input_section = input_sections[in_sec_id]

        print("before structure filling")

        # Load the structure of the english text
        if isinstance(input_section, SectionStructure):
            structure = input_section
        else:
            structure.fill(input_section) 

        if args.outsource_rhyme_schemes and args.from_dict:
            structure.rhyme_scheme = espeak_rhymes[in_sec_id]

        print("after structure filling")

        prompt = prepare_prompt_whole(dataset_type, structure)
        print(prompt)

        inputs = tokenizer([prompt],return_token_type_ids=False, return_tensors="pt").to(device)

        print("before generation")

        sample_outputs = model.generate(**inputs,
            do_sample=True,
            top_p=0.95,
            repetition_penalty=1.0,
            temperature=0.8,
            max_new_tokens=256,
            num_return_sequences=args.out_per_gerenation,
            pad_token_id=tokenizer.eos_token_id,
            penalty_alpha=0.6,
            stopping_criteria=StoppingSequenceCriteria(prompt, tokenizer, structure.num_lines),
            )
        
        print("after generation")
        
        out_lyrics = []
        for sample_output in sample_outputs:
            out_lyrics.append(extract_output_whole(tokenizer.decode(sample_output.tolist(), skip_special_tokens=True), prompt, dataset_type, structure))
        
        print("before postprocessing")
        model_out = postprocesser.choose_best_section(out_lyrics, structure, remove_add_stopwords=args.postprocess_stopwords)
        print("after postprocessing")
        
        print(f"\n{', '.join(model_out)}\n")

        result_pairs.append((','.join(model_out), structure.copy()))
        for line in model_out:
            print(line)
        print()

    return result_pairs