import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, StoppingCriteria
from dataset_types import DatasetType
from english_structure_extractor import SectionStructure
import os



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

    model_path = os.path.join(args.model_path, f"{args.model}_{dataset_type.name}_lyricist_{args.epoch}_32.pt")

    print("="*10 + "  " + model_path + " " + "="*10)
    model.load_state_dict(state_dict=torch.load(model_path, map_location=torch.device(device)))
    model.eval()

    structure = SectionStructure()

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
            num_return_sequences=5,
            penalty_alpha=0.6,
            max_new_tokens=512,
            stopping_criteria=StoppingSequenceCriteria(prompt, tokenizer),
            )

        # generated sequence
        for i, sample_output in enumerate(sample_outputs):
            model_out = tokenizer.decode(sample_output.tolist(), skip_special_tokens=True)
            print("\n{}\n\n{}\n".format(i+1, model_out)) # tokenizer.decode(sample_output, skip_special_tokens=True)

            model_output = model_output.split("\n")
            start_of_text = 1
            if dataset_type == DatasetType.BASELINE:
                start_of_text = 0

            for line_i in range(start_of_text, len(model_output)):
                line = model_output[line_i]
                if not line.strip():
                    continue
                line = line.split("#")[-1].strip()
                if not line.strip():
                    continue
                model_output[line_i] = line

            result_pairs.append((','.join(model_output), structure.copy()))
            for line in model_output:
                print(line)
            print()

    return result_pairs