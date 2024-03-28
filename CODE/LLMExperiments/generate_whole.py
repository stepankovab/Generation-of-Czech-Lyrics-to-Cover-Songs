import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, StoppingCriteria
from dataset_types import DatasetType
from english_structure_extractor import SectionStructure
from postprocessing import Postprocesser
import os
import re



class StoppingSequenceCriteria(StoppingCriteria):
    def __init__(self, prompt, tokenizer):
        self.prompt=prompt
        self.tokenizer = tokenizer
        self.number_of_lines = len(prompt.split("#")[0].strip().split(" "))

    def __call__(self, input_ids, scores, **kwargs):
        generated_text = self.tokenizer.decode(input_ids[0])
        generated_text = generated_text.replace(self.prompt,'')
        generated_lines = generated_text.strip().split("\n")
        if len(generated_lines) > self.number_of_lines:
            return True  # Stop generation
        return False  # Continue generation

    def __len__(self):
        return 1

    def __iter__(self):
        yield self


def prepare_prompt(dataset_type, structure: SectionStructure):
    if dataset_type == DatasetType.BASELINE:
            prompt = ""
    elif dataset_type == DatasetType.SYLLABLES:
        prompt = f"{' '.join([str(x) for x in structure.syllables])} #\n"
    elif dataset_type == DatasetType.CHARACTERISTIC_WORDS:
        prompt = f"{structure.num_lines} # {' '.join(structure.keywords)} #\n"
    elif dataset_type == DatasetType.SYLLABLES_AND_WORDS:
        prompt = f"{' '.join([str(x) for x in structure.syllables])} # {' '.join(structure.keywords)} #\n"
    elif dataset_type == DatasetType.FORCED_SYLLABLES:
        prompt = f"{' '.join([str(x) for x in structure.syllables])} #\n"
    else:
        raise Exception(f"We don't support a Dataset type {dataset_type}")

    return prompt


def extract_model_out(out_lines, prompt, dataset_type, structure):
    out_lines = out_lines.split("\n")
    start_of_text = 1
    if not prompt.strip():
        start_of_text = 0

    output = []
    for line_i in range(start_of_text, min(len(out_lines), structure.num_lines + start_of_text)):
        line = out_lines[line_i]
        line = re.sub(',', '', line)
        if not line.strip():
            output.append("")
            continue
    
        line_sections = line.strip().split("#")
        if dataset_type in [DatasetType.SYLLABLES, DatasetType.CHARACTERISTIC_WORDS, DatasetType.SYLLABLES_AND_WORDS, DatasetType.FORCED_SYLLABLES]:
            if len(line_sections) > 1:
                line = ' # '.join([x.strip() for x in line_sections[1:]])

        output.append(line)
        
    return output

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

    elif args.model == "Mistral_czech":
        tokenizer = AutoTokenizer.from_pretrained("simecek/cswikimistral_0.1")
        model = AutoModelForCausalLM.from_pretrained("simecek/cswikimistral_0.1")
        tokenizer.model_max_length=1024

    elif args.model == "tinyLlama":
        tokenizer = AutoTokenizer.from_pretrained("TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T")
        model = AutoModelForCausalLM.from_pretrained("TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T")
        tokenizer.model_max_length=1024

    model_path = os.path.join(args.model_path, f"{args.model}_{dataset_type.name}_{args.generation_method}_{args.epoch}.pt")

    print("="*10 + "  " + model_path + " " + "="*10)
    model.load_state_dict(state_dict=torch.load(model_path, map_location=torch.device(device)))
    model.eval()

    structure = SectionStructure()
    postprocesser = Postprocesser()

    result_pairs = []

    for input_section in input_sections:
        # Load the structure of the english text
        structure.fill(input_section) 

        prompt = prepare_prompt(dataset_type, structure)
        print(prompt)

        inputs = tokenizer(prompt, return_tensors="pt") 
        tokenizer.encode(prompt, return_tensors="pt") #directly for input_ids

        # model output using Top-k sampling text generation method
        sample_outputs = model.generate(**inputs,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
            num_return_sequences=args.out_per_gerenation,
            penalty_alpha=0.6,
            max_new_tokens=512,
            stopping_criteria=StoppingSequenceCriteria(prompt, tokenizer),
            )
        
        out_lyrics = []
        for sample_output in sample_outputs:
            out_lyrics.append(extract_model_out(tokenizer.decode(sample_output.tolist(), skip_special_tokens=True), prompt, dataset_type, structure))
        model_out = postprocesser.choose_best_section(out_lyrics, structure)
        
        print(f"\n{', '.join(model_out)}\n")

        result_pairs.append((','.join(model_out), structure.copy()))
        for line in model_out:
            print(line)
        print()

    return result_pairs