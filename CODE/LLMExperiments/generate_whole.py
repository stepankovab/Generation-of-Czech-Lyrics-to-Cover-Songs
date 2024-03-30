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
    
    dataset_type = DatasetType(args.dataset_type)
    
    device = 'cpu'
    if torch.cuda.is_available():
        print("cuda available.")
        device = 'cuda'

    if args.model == "GPT2_oscar":
        tokenizer = AutoTokenizer.from_pretrained("lchaloupsky/czech-gpt2-oscar")
        model = AutoModelForCausalLM.from_pretrained("lchaloupsky/czech-gpt2-oscar")
        tokenizer.model_max_length=1024

    elif args.model == "GPT2_czech_XL":
        tokenizer = AutoTokenizer.from_pretrained("BUT-FIT/Czech-GPT-2-XL-133k")
        model = AutoModelForCausalLM.from_pretrained("BUT-FIT/Czech-GPT-2-XL-133k")
        tokenizer.model_max_length=1024

    elif args.model == "tinyLlama":
        tokenizer = AutoTokenizer.from_pretrained("TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T")
        model = AutoModelForCausalLM.from_pretrained("TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T")
        tokenizer.model_max_length=1024
    
    else:
        raise ValueError(f"Model {args.model} is not supported with generation mode 'whole'.")

    model_path = os.path.join(args.model_path, f"{args.model}_{dataset_type.name}_{args.generation_method}_{args.epoch}.pt")

    print("="*10 + "  " + model_path + " " + "="*10)
    model.load_state_dict(state_dict=torch.load(model_path, map_location=torch.device(device)))
    model.eval()

    structure = SectionStructure(rt=RhymerType(args.rhymer))
    postprocesser = Postprocesser(evaluator=Evaluator(rt=RhymerType(args.rhymer)))

    result_pairs = []

    for input_section in input_sections:
        print("before structure filling")
        # Load the structure of the english text
        structure.fill(input_section) 

        print("after structure filling")

        prompt = prepare_prompt_whole(dataset_type, structure)
        print(prompt)

        inputs = tokenizer(prompt, return_tensors="pt") 
        tokenizer.encode(prompt, return_tensors="pt") #directly for input_ids

        print("before generation")

        # model output using Top-k sampling text generation method
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