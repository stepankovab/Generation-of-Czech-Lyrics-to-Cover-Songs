from evaluator import Evaluator
from HT_loader import HT_loader
import argparse
from generate_whole import generate_whole
from generate_lines import generate_lines


parser = argparse.ArgumentParser()
parser.add_argument("--batch_size", default=32, type=int, help="Batch size")
parser.add_argument("--model", default="GPT2_oscar", type=str, help="tinyLlama # GPT2_oscar # GPT2_czech_XL # Mistral_czech # ") 
parser.add_argument("--epoch", default=3, type=int, help="Epoch of the trained model")
parser.add_argument("--dataset_path", default="./", type=str, help="./ # DATA/Velky_zpevnik ")
parser.add_argument("--model_path", default="./CODE/LLMExperiments/trained_models", type=str, help="./ #  CODE/LLMExperiments/trained_models")
parser.add_argument("--input_section", default="let it go,let it go,can't hold it back anymore,turn away and slam the door", type=str, help="Input section in English")
parser.add_argument("--from_dict", default=False, type=bool, help="Take testing data from HT dict")
parser.add_argument("--dataset_type", default=6, type=int, help="Dataset type: BASELINE = 1, SYLLABLES = 2, END_OF_LINES = 3, CHARACTERISTIC_WORDS = 4, UNRHYMED = 5, SYLLABLES_AND_WORDS = 6, SYLLABLES_AND_ENDS = 7, ENDS_AND_WORDS = 8")
parser.add_argument("--generation_method", default="whole", type=str, help="whole, lines")
parser.add_argument("--out_per_gerenation", default=10, type=int, help="number of generated outputs for preprocessing")
args = parser.parse_args([] if "__file__" not in globals() else None)

########### get input data ###########
if args.from_dict == True:
    input_sections = HT_loader(args.dataset_path)
else:
    input_sections = [args.input_section, "this is a test, so try your best"]

########### generate outputs #############
if args.generation_method == "whole":
    result_pairs = generate_whole(args, input_sections)
elif args.generation_method == "lines":
    result_pairs = generate_lines(args, input_sections)

########### evaluate outputs ##############
evaluator = Evaluator()
results_dict = evaluator.evaluate_outputs_structure(result_pairs)


for lyrics, structure in result_pairs:
    in_lyrics = '\n'.join(structure.original_lyrics)
    out_lyrics = '\n'.join(lyrics.split(','))
    print(f"======================================\n\n{in_lyrics}\n\n{out_lyrics}\n")
    

for cat in results_dict:
    if len(results_dict[cat]) == 0:
        continue
    print(f"{cat} -> {sum(results_dict[cat]) / len(results_dict[cat])}")
print()
print("=" * 30)