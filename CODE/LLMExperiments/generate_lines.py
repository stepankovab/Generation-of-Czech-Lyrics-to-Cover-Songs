import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, StoppingCriteria
from dataset_types import DatasetType
from english_structure_extractor import SectionStructure
from CODE.LLMExperiments.evaluator import Evaluator
from eval.syllabator import syllabify


class StoppingSequenceCriteria(StoppingCriteria):
    def __init__(self, prompt, tokenizer):
        self.prompt=prompt
        self.tokenizer = tokenizer
        self.number_of_lines = len(prompt.split("#")[0].strip().split(" "))

    def __call__(self, input_ids, scores, **kwargs):
        generated_text = self.tokenizer.decode(input_ids[0])
        generated_text = generated_text.replace(self.prompt,'')
        generated_lines = generated_text.strip().split("\n")
        if (len(generated_lines) == self.number_of_lines and len(syllabify(generated_lines[-1].split("#")[-1].strip())) >= int(self.prompt.split("#")[0].strip().split(" ")[-1])) or len(generated_lines) > self.number_of_lines:
            return True  # Stop generation
        return False  # Continue generation

    def __len__(self):
        return 1

    def __iter__(self):
        yield self


def generate_lines(model, model_path, tokenizer, input_sections, dataset_type, device="cpu"):
    model.load_state_dict(state_dict=torch.load(model_path, map_location=torch.device(device)))
    model.eval()

    for input_section in input_sections:
        # Load the structure of the english text
        og_structure = SectionStructure()
        og_structure.fill(input_section) 

        if dataset_type == DatasetType.BASELINE:
            prompt = ""

        elif dataset_type == DatasetType.SYLLABLES:
            prompt = f"{' '.join([str(x) for x in og_structure.syllables])} #\n"
                    
        elif dataset_type == DatasetType.END_OF_LINES:
            prompt = ""

        elif dataset_type == DatasetType.CHARACTERISTIC_WORDS:
            prompt = f"{og_structure.num_lines} # {' '.join(og_structure.keywords)} #\n"

        elif dataset_type == DatasetType.UNRHYMED_LEN:
            pass

        elif dataset_type == DatasetType.SYLLABLES_AND_WORDS:
            prompt = f"{' '.join([str(x) for x in og_structure.syllables])} # {' '.join(og_structure.keywords)} #\n"

        elif dataset_type == DatasetType.SYLLABLES_AND_ENDS:
            prompt = f"{' '.join([str(x) for x in og_structure.syllables])} # ví nám ní ky #\n"

        elif dataset_type == DatasetType.ENDS_AND_WORDS:
            prompt = f"ví nám tel sám # {' '.join(og_structure.keywords)} #\n"

        elif dataset_type == DatasetType.FORCED_SYLLABLES:
            prompt = f"{' '.join([str(x) for x in og_structure.syllables])} #\n"

        else:
            raise Exception(f"We don't support a Dataset type {dataset_type}")

        # print(prompt)

        inputs = tokenizer(prompt, return_tensors="pt") 
        tokenizer.encode(prompt, return_tensors="pt") #directly for input_ids

        # model output using Top-k sampling text generation method
        sample_outputs = model.generate(**inputs,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
            num_return_sequences=3,
            penalty_alpha=0.6,
            max_new_tokens=512,
            stopping_criteria=StoppingSequenceCriteria(prompt, tokenizer),
            )

        if dataset_type == DatasetType.SYLLABLES or dataset_type == DatasetType.SYLLABLES_AND_WORDS:
            evaluator = Evaluator()

        avg_length_ratio = 0
        avg_syll_distance = 0
        avg_syll_accuracy = 0
        avg_end_accuracy = 0
        avg_keyword_similarity = 0

        # generated sequence
        for i, sample_output in enumerate(sample_outputs):
            model_out = tokenizer.decode(sample_output.tolist(), skip_special_tokens=True)
            print("\n{}\n\n{}\n".format(i+1, model_out)) # tokenizer.decode(sample_output, skip_special_tokens=True)

            if dataset_type == DatasetType.SYLLABLES or dataset_type == DatasetType.SYLLABLES_AND_WORDS:
                length_ratio, syll_distance, syll_accuracy, end_accuracy, keyword_similarity = evaluator.eval_from_first_line(model_out)
                print(f"length_ratio = {length_ratio}")
                print(f"syll_distance = {syll_distance}")
                print(f"syll_accuracy = {syll_accuracy}")
                print(f"keyword_similarity = {keyword_similarity}")
                # print(f"end_accuracy = {end_accuracy}")

                avg_length_ratio += length_ratio
                avg_syll_distance += syll_distance 
                avg_syll_accuracy += syll_accuracy
                # avg_end_accuracy += end_accuracy
                avg_keyword_similarity += keyword_similarity


    print("="*100)
    print("\n\nepoch average:\n")
    print(f"length_ratio = {avg_length_ratio / len(sample_outputs)}")
    print(f"syll_distance = {avg_syll_distance / len(sample_outputs)}")
    print(f"syll_accuracy = {avg_syll_accuracy / len(sample_outputs)}")
    print(f"keyword_similarity = {avg_keyword_similarity / len(sample_outputs)}")
    # print(f"end_accuracy = {avg_end_accuracy / len(sample_outputs)}")
    print("="*100)
    print("\n")

