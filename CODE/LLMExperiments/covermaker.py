from evaluator import Evaluator
from rhymer_types import RhymerType
from english_structure_extractor import SectionStructure
from HT_loader import HT_loader
from generate_whole import generate_whole
from generate_lines import generate_lines
from fewshot import fewshot_and_generate
import time
import json
import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--model", default="OSCAR_GPT2", type=str, help="OSCAR_GPT2 # VUT_GPT2 # TINYLLAMA # VUT_TINYLLAMA") 
parser.add_argument("--model_path", default="C:\\Users\\barca\\MOJE\\BAKALARKA\\trained_models", type=str, help="./ #  CODE/LLMExperiments/trained_models")
parser.add_argument("--input_section", default="You are the dancing queen,Young and sweet,Only seventeen,Dancing queen,Feel the beat from the tambourine;You can dance,You can jive,Having the time of your life", type=str, help="Input section in English, lines divided by comma ',' and sections divided by semicolon ';' -> eg: let it go,let it go,can't hold it back anymore,turn away and slam the door")
parser.add_argument("--dataset_type", default=5, type=int, help="Dataset type: BASELINE = 1, SYLLABLES = 2, END_OF_LINES = 3, CHARACTERISTIC_WORDS = 4, UNRHYMED = 5, SYLLABLES_AND_WORDS = 6, SYLLABLES_AND_ENDS = 7, ENDS_AND_WORDS = 8")
parser.add_argument("--from_dict", default=False, type=bool, help="Take testing data from HT dict")
parser.add_argument("--test_set_size", default=1, type=int, help="How many samples from test set to take, 0 means all")
parser.add_argument("--dataset_path", default="c:\\Users\\barca\\MOJE\\BAKALARKA", type=str, help="./ # DATA/Velky_zpevnik ")
parser.add_argument("--epoch", default=0, type=int, help="Epoch of the trained model")
parser.add_argument("--generation_method", default="whole", type=str, help="whole, lines, fewshot")
parser.add_argument("--out_per_generation", default=10, type=int, help="number of generated outputs to choose the best from")
parser.add_argument("--nshot", default=10, type=int, help="Number of examples when using few-shot as generation method")
parser.add_argument("--rhymer", default=3, type=int, help="Rhymer type: RHYMETAGGER = 1, RHYMEFINDER = 2, SAME_WORD_RHYMETAGGER = 3")
parser.add_argument("--postprocess_stopwords", default=True, type=bool, help="Posrprocess each output by trying to correct the length by removing/adding stopwords")
parser.add_argument("--results_path", default="./results_dicts", type=str, help="path to folder to save the results")
parser.add_argument("--load_structures", default=True, type=str, help="take rhymes from outside")
args = parser.parse_args([] if "__file__" not in globals() else None)

########### get input data ###########
if args.from_dict and not args.load_structures:
    input_sections = HT_loader(args.dataset_path, language="en")
    if args.test_set_size > 0:
        input_sections = input_sections[:min(args.test_set_size, len(input_sections))]

elif args.from_dict and args.load_structures:
    with open(os.path.join(args.dataset_path, "english_structure_list.json"), "r", encoding='utf-8') as json_file:
        input_sections_dicts_list = json.load(json_file)

    input_sections = []
    if args.test_set_size == 0:
        args.test_set_size = len(input_sections_dicts_list)
    for section_id in range(min(len(input_sections_dicts_list), args.test_set_size)):
        section = SectionStructure()
        section.fill_from_dict(input_sections_dicts_list[section_id])
        input_sections.append(section)
else:
    input_sections = args.input_section.split(';')

########### generate outputs #############
gen_start_stamp = time.time()
if args.generation_method == "whole":
    result_pairs = generate_whole(args, input_sections, verbose=False)
elif args.generation_method == "lines":
    result_pairs = generate_lines(args, input_sections, verbose=False)
elif args.generation_method == "fewshot":
    result_pairs = fewshot_and_generate(args, input_sections, verbose=False)
gen_end_stamp = time.time()

########### evaluate outputs ##############
eval_start_stamp = time.time()
evaluator = Evaluator(verbose=False, czech_rhyme_detector=RhymerType(args.rhymer))
results_dict = evaluator.evaluate_outputs_structure(result_pairs, evaluate_keywords=True, evaluate_line_keywords=True, evaluate_translations=True)
eval_stop_stamp = time.time()

########### print results ###############
for lyrics, structure in result_pairs:
    in_lyrics = '\n'.join(structure.original_lyrics_list)
    out_lyrics = '\n'.join(lyrics.split(','))
    print(f"{'=' * 30}\n\n{in_lyrics}\n\n{out_lyrics}\n")
    
print("=" * 30)
print()
for cat in results_dict:
    if len(results_dict[cat]) == 0:
        continue
    print(f"{cat} -> {round(sum(results_dict[cat]) / len(results_dict[cat]), 2)}")
print()
print("=" * 30)
print()
print(f"eval time: {(eval_stop_stamp - eval_start_stamp) / 60:2f}min")
if len(input_sections) > 0:
    print(f"generation time: {(gen_end_stamp - gen_start_stamp)/60:2f}min, with {((gen_end_stamp - gen_start_stamp)) / len(input_sections):2f}s per lyrics section")

if args.from_dict == True:
    lyric_pairs = []
    for lyrics, structure in result_pairs:
        in_lyrics = structure.original_lyrics_list
        out_lyrics = lyrics.split(',')
        lyric_pairs.append((in_lyrics, out_lyrics))
    results_dict["lyrics"] = lyric_pairs

    if not os.path.exists(args.results_path):
        os.mkdir(args.results_path)

    print("==================== dumping results into json =====================")
    with open(os.path.join(args.results_path, f"{args.model}_{args.generation_method}_dataset_type_{args.dataset_type}_epoch_{args.epoch}_samples_{args.test_set_size}_out_per_generation_{args.out_per_generation}_stopwords_{args.postprocess_stopwords}_rhymer_{args.rhymer}_fewshot_{args.nshot}.json"), "w", encoding='utf-8') as json_file:
        json.dump(results_dict, json_file, ensure_ascii=False)

    print(f"{args.model}_{args.generation_method}_dataset_type_{args.dataset_type}_epoch_{args.epoch}_samples_{args.test_set_size}_out_per_generation_{args.out_per_generation}_stopwords_{args.postprocess_stopwords}_rhymer_{args.rhymer}_fewshot_{args.nshot}")
