import torch
import logging
import random
from lyrics_datasets import WholeLyricsDataset, prepare_prompt_whole, extract_output_whole
from transformers import AutoTokenizer, AutoModelForCausalLM, StoppingCriteria
from dataset_types import DatasetType
from rhymer_types import RhymerType
from english_structure_extractor import SectionStructure
from postprocessing import Postprocesser
from evaluator import Evaluator
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
   

def get_n_examples(dataset_path, dataset_type, n):
    a = WholeLyricsDataset(dataset_path, dataset_type)
    random.seed(31)
    random.shuffle(a.lyrics_list)
    return '\n'.join(a.lyrics_list[:min(n, len(a.lyrics_list))]) + "\n"


def fewshot_and_generate(args, input_sections):
    dataset_type = DatasetType(args.dataset_type)
    
    device = 'cpu'
    if torch.cuda.is_available():
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
    print(f"loaded model: {args.model}")

    # Set special tokens if they are not already set
    if tokenizer.sep_token is None:
        tokenizer.add_special_tokens({'sep_token': '[SEP]'})
    if tokenizer.cls_token is None:
        tokenizer.add_special_tokens({'cls_token': '[CLS]'})
    if tokenizer.mask_token is None:
        tokenizer.add_special_tokens({'mask_token': '[MASK]'})

    n_examples = get_n_examples(args.dataset_path, dataset_type, args.nshot)
    print(n_examples)
    
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
        prompt = n_examples + prompt


        inputs = tokenizer([prompt],return_token_type_ids=False, return_tensors="pt").to(device)
    
        logging.info("prompt length: ", inputs["input_ids"].shape)
        output_ids = model.generate(**inputs,
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
        
        outputs = []
        for o_ids in output_ids:
            outputs.append(extract_output_whole(tokenizer.decode(o_ids.tolist(), skip_special_tokens=True), prompt, dataset_type, structure))
        print(outputs)
        print("after generation")
        
        print("before postprocessing")
        model_out = postprocesser.choose_best_section(outputs, structure, remove_add_stopwords=args.postprocess_stopwords)
        print("after postprocessing")
        
        print(f"\n{', '.join(model_out)}\n")

        result_pairs.append((','.join(model_out), structure.copy()))
        for line in model_out:
            print(line)
        print()

    return result_pairs






# simecek/cswikimistral_0.1
# BUT-FIT/Czech-GPT-2-XL-133k
# lchaloupsky/czech-gpt2-oscar
# TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T
# BUT-FIT/CSTinyLlama-1.2B
# BUT-FIT/csmpt7b
# mistralai/Mistral-7B-v0.1










# def get_3_shot_paragraph_prompt(sylls: str, keywords: list, ending: str):
# #     return """
# # ### System: Konverzace mezi člověkem a automatickým asistentem. Asistent pomáhá člověku vymýšlet větu o jednom řádku, která se rýmuje s poksytnutým slovem. Věta obsahuje poskytnutý počet slabik. Navíc asistent vymyslí větu co dává smysl podle poskytnutých klíčových slov.
# # ### Člověk: Rým = pes, počet slabik = 9, klíčová slova = les voda
# # ### Asistent: Za lesem u vody leží ves.
# # ### Člověk: Rým = důl, počet slabik = 5, klíčová slova = stůl jídlo
# # ### Asistent: Na stole je sůl.
# # ### Člověk: Rým = vaz, počet slabik = 7, klíčová slova = zuby problém
# # ### Asistent: A zase mám další kaz.
# # ### Člověk: {}
# # ### Asistent: 
# # """.format(question)
#     return """
# ### System: Konverzace mezi člověkem a automatickým asistentem. Asistent pomáhá člověku vymýšlet větu o jednom řádku, která se rýmuje s poksytnutým slovem. Věta obsahuje poskytnutý počet slabik. Navíc asistent vymyslí větu co dává smysl podle poskytnutých klíčových slov.
# ### Člověk: Napiš řádku na 9 slabik o 'les, voda', ať končí na 'ves'
# ### Asistent: Za lesem u vody leží ves.
# ### Člověk: Napiš řádku na 5 slabik o 'stůl, jídlo', ať končí na 'důl'
# ### Asistent: Na stole je sůl.
# ### Člověk: Napiš řádku na 7 slabik o 'zuby, problém', ať končí na 'vaz'
# ### Asistent: A zase mám další kaz.
# ### Člověk: Napiš řádku na {} slabik o '{}', ať končí na '{}'
# ### Asistent: 
# """.format(sylls, ', '.join(keywords), ending)