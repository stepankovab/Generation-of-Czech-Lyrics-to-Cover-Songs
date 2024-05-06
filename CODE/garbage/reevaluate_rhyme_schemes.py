import json
import os
from evaluator import Evaluator
from eval.same_word_tagger import SameWordRhymeTagger
from english_structure_extractor import SectionStructure

ev = Evaluator(czech_rhyme_detector=SameWordRhymeTagger("cs"))
structure = SectionStructure()

with open("english_structure_list.json", mode="r", encoding="utf-8") as json_file:
    en_lyrics = json.load(json_file)

rhyme_agree = []
rhyme_acc = []

for filename in os.listdir("CODE\\LLMExperiments\\results_dicts_meta"):
    print(filename)

    with open(os.path.join("CODE\\LLMExperiments\\results_dicts_meta", filename), "r", encoding="utf-8") as json_file:
        results_dict = json.load(json_file)

    for i in range(len(results_dict["lyrics"])):
        structure.fill_from_dict(en_lyrics[i])
        cs_lyrics = results_dict["lyrics"][i][1]
        cs_scheme = ev.czech_rhyme_detector.tag(poem=cs_lyrics, output_format=3)

        rhyme_agree.append(ev.get_rhyme_scheme_agreement(structure.rhyme_scheme, cs_scheme))
        rhyme_acc.append(ev.get_rhyme_scheme_accuracy(structure.rhyme_scheme, cs_scheme))

    print(f"agree -> {sum(rhyme_agree) / len(rhyme_agree)}")
    print(f"acc -> {sum(rhyme_acc) / len(rhyme_acc)}")
    print()



        

     


