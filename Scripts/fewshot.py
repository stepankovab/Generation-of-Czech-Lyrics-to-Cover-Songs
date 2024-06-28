import torch
import logging
import random
from lyrics_datasets import WholeLyricsDataset, prepare_prompt_whole, extract_output_whole
from transformers import AutoTokenizer, AutoModelForCausalLM, StoppingCriteria
from prompt_types import PromptType
from rhymer_types import RhymerType
from english_structure_extractor import SectionStructureExtractor, SectionStructure
from postprocessing import Postprocesser
from evaluator import Evaluator

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
   
def get_n_examples(dataset_path, dataset_type, n):
    a = WholeLyricsDataset(dataset_path, dataset_type)
    random.seed(31)
    random.shuffle(a.lyrics_list)
    return '\n'.join(a.lyrics_list[:min(n, len(a.lyrics_list))]) + "\n"

def fewshot_and_generate(args, input_sections, verbose=False):
    dataset_type = PromptType(args.prompt_type)
    
    device = 'cpu'
    if torch.cuda.is_available():
        if verbose:
            print("cuda available.")
        device = 'cuda'

    if args.model == "OSCAR_GPT2":
        args.model = "lchaloupsky/czech-gpt2-oscar"
    elif args.model == "VUT_GPT2":
        args.model = "BUT-FIT/Czech-GPT-2-XL-133k"
    elif args.model == "TINYLLAMA":
        args.model = "TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T"
    elif args.model == "VUT_TINYLLAMA":
        args.model = "BUT-FIT/CSTinyLlama-1.2B"

    model, tokenizer = AutoModelForCausalLM.from_pretrained(args.model), AutoTokenizer.from_pretrained(args.model)
    model.to(device)
    if verbose:
        print(f"loaded model: {args.model}")

    # Set special tokens if they are not already set
    if tokenizer.sep_token is None:
        tokenizer.add_special_tokens({'sep_token': '[SEP]'})
    if tokenizer.cls_token is None:
        tokenizer.add_special_tokens({'cls_token': '[CLS]'})
    if tokenizer.mask_token is None:
        tokenizer.add_special_tokens({'mask_token': '[MASK]'})

    n_examples = get_n_examples(args.dataset_path, dataset_type, args.nshot)
    
    structure_extractor = SectionStructureExtractor(english_rhyme_detector=RhymerType(args.rhymer))
    postprocesser = Postprocesser(evaluator=Evaluator(czech_rhyme_detector=RhymerType(args.rhymer)))

    result_pairs = []

    for in_sec_id in range(len(input_sections)):
        input_section = input_sections[in_sec_id]

        if isinstance(input_section, SectionStructure):
            structure = input_section
        else:
            structure = structure_extractor.create_section_structure(input_section) 

        prompt = prepare_prompt_whole(dataset_type, structure)
        prompt = n_examples + prompt


        inputs = tokenizer([prompt],return_token_type_ids=False, return_tensors="pt").to(device)
    
        logging.info("prompt length: ", inputs["input_ids"].shape)
        output_ids = model.generate(**inputs,
            do_sample=True,
            top_p=0.95,
            repetition_penalty=1.0,
            temperature=0.8,
            max_new_tokens=256,
            num_return_sequences=args.choose_best,
            pad_token_id=tokenizer.eos_token_id,
            penalty_alpha=0.6,
            stopping_criteria=StoppingSequenceCriteria(prompt, tokenizer, structure.num_lines),
            )
        
        outputs = []
        for o_ids in output_ids:
            outputs.append(extract_output_whole(tokenizer.decode(o_ids.tolist(), skip_special_tokens=True), prompt, dataset_type, structure))
        model_out = postprocesser.choose_best_section(outputs, structure, remove_add_stopwords=args.postprocess_stopwords)

        if verbose:
            print(f"\n{', '.join(model_out)}\n")

        result_pairs.append((','.join(model_out), structure))

    return result_pairs
