import argparse
from generate_whole import generate_whole
import sys

parser = argparse.ArgumentParser()
parser.add_argument("--model", default="OSCAR_GPT2", type=str, help="OSCAR_GPT2 # VUT_GPT2 # TINYLLAMA # VUT_TINYLLAMA") 
parser.add_argument("--model_path", default="C:\\Users\\barca\\MOJE\\BAKALARKA\\trained_models", type=str, help="./ #  CODE/LLMExperiments/trained_models")
parser.add_argument("--input_section", default="You are the dancing queen,Young and sweet,Only seventeen,Dancing queen,Feel the beat from the tambourine;You can dance,You can jive,Having the time of your life", type=str, help="Input section in English, lines divided by comma ',' and sections divided by semicolon ';' -> eg: let it go,let it go,can't hold it back anymore,turn away and slam the door")
parser.add_argument("--prompt_type", default=5, type=int, help="Dataset type: BASELINE = 1, SYLLABLES = 2, END_OF_LINES = 3, CHARACTERISTIC_WORDS = 4, UNRHYMED = 5, SYLLABLES_AND_WORDS = 6, SYLLABLES_AND_ENDS = 7, ENDS_AND_WORDS = 8")
parser.add_argument("--generation_method", default="whole", type=str, help="whole, lines, fewshot")
parser.add_argument("--choose_best", default=10, type=int, help="number of generated outputs to choose the best from")
parser.add_argument("--rhymer", default=1, type=int, help="The rhyme detector to be used. Options: RHYMEFINDER = 1, RHYMETAGGER = 2, SAME_WORD_RHYMETAGGER = 3")
parser.add_argument("--postprocess_stopwords", default=True, type=bool, help="Posrprocess each output by trying to correct the length by removing/adding stopwords")
args = parser.parse_args([] if "__file__" not in globals() else None)

########### get input data ###########
input_sections = args.input_section.split(';')

########### generate outputs #############
result_pairs = generate_whole(args, input_sections, verbose=False)

########### print results ###############
sys.stdin.reconfigure(encoding='utf-8')
sys.stdout.reconfigure(encoding='utf-8')

for lyrics, structure in result_pairs:
    print(lyrics)
   