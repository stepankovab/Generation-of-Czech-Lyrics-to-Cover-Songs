from HT_loader import HT_loader
from evaluator import Evaluator
from english_structure_extractor import SectionStructureExtractor, SectionStructure
from eval.rhyme_finder import RhymeFinder
from eval.tagger import RhymeTagger
from eval.same_word_tagger import SameWordRhymeTagger
from eval.syllabator import syllabify as cs_syll
from eval.en_syllabator import syllabify as en_syll
from rhymer_types import RhymerType
import matplotlib.pyplot as plt
import random
import numpy as np
import os
import re
import json

RANDOM_BASELINE = False
HT = True

evaluator = Evaluator(czech_rhyme_detector=RhymerType.SAME_WORD_RHYMETAGGER)
# structureExtractor = SectionStructureExtractor(kw_model=evaluator.kw_model, english_rhyme_detector=RhymerType.SAME_WORD_RHYMETAGGER)
# rt_cs=SameWordRhymeTagger("cs")
# rt_en=SameWordRhymeTagger("en")

cs_lyrics = HT_loader("./", "cs")

# with open(os.path.join("CODE","LLMExperiments","outs_eval", "outputs.o21441454"), mode="r", encoding="utf-8") as f:
#     lines = f.readlines()

# cs_lyrics = []
# for i in range(len(lines)):
#     line = lines[i]
#     if re.match("after postprocessing", line):
#         cs_lyrics.append(",".join(lines[i+2].strip().split(", ")))

with open("english_structure_list.json", mode="r", encoding="utf-8") as json_file:
    en_lyrics = json.load(json_file)

# with open("czech_machine_translations_list.json", mode="r", encoding="utf-8") as json_file:
#     cs_lyrics = json.load(json_file)


# en_lyrics = HT_loader("./", "en")

if RANDOM_BASELINE:
    random.shuffle(en_lyrics)

assert len(en_lyrics) == len(cs_lyrics)

HT_pairs = []
# results_dict = {"syll_dist" : []
#                 }

for i in range(len(en_lyrics)):
    structure = SectionStructure()
    structure.fill_from_dict(en_lyrics[i])

    # cs_split = cs_lyrics[i].split(',')
    # en_split = en_lyrics[i].split(',')

    # cs_scheme = rt_cs.tag(cs_split)
    # en_scheme = rt_en.tag(en_split)

    # agreement_cs = evaluator.get_rhyme_scheme_agreement(en_scheme, cs_scheme)
    # accuracy = evaluator.get_rhyme_scheme_accuracy(cs_scheme, en_scheme)

    # results_dict["rhyme_scheme_agree"].append(agreement_cs)
    # results_dict["rhyme_accuracy"].append(accuracy)

    if RANDOM_BASELINE:
        cs_split = cs_lyrics[i].split(',')

        if len(cs_split) == structure.num_lines:
            continue

        elif len(cs_split) < structure.num_lines:
            structure.en_line_keywords = structure.en_line_keywords[:len(cs_split)]
            structure.line_keywords = structure.line_keywords[:len(cs_split)]
            structure.num_lines = len(cs_split)
            structure.original_lyrics_list = structure.original_lyrics_list[:len(cs_split)]
            structure.rhyme_scheme = structure.rhyme_scheme[:len(cs_split)]
            structure.syllables = structure.syllables[:len(cs_split)]
        
        else:
            cs_lyrics[i] = ','.join(cs_split[:structure.num_lines])
    
    # en_sylls = []
    # for i in range(len(en_split)):
    #     en_sylls.append(len(en_syll(en_split[i])))
    
    # cs_sylls = []
    # for i in range(len(cs_split)):
    #     cs_sylls.append(len(cs_syll(cs_split[i])))

    # results_dict["syll_dist"].append(evaluator.get_section_syllable_distance(cs_sylls, structure.syllables))

    
    # structure = structureExtractor.create_section_structure(en_lyrics[i])
    HT_pairs.append((cs_lyrics[i], structure))
    
results_dict = evaluator.evaluate_outputs_structure(HT_pairs, evaluate_keywords=True, evaluate_line_keywords=True, evaluate_translations=True)

# with open("baseline_results_dict.json", "r", encoding="utf-8") as json_file:
#     results_dict = json.load(json_file)

name_dict = {"syll_dist" : "Syll. Distance",
            "syll_acc" : "Syll. Accuracy",
            "rhyme_scheme_agree" : "Rhyme Agreement",
            "rhyme_accuracy" : "Rhyme Accuracy",
            "semantic_sim" : "Semantic Similarity",
            "keyword_sim" : "Keyword Similarity",
            "line_keyword_sim" : "Line-by-line Sim.",
            "phon_rep_dif" : "Phoneme Rep. Dif.",
            "bleu4gram" : "BLEU (4-gram)",
            "bleu2gram" : "BLEU (2-gram)",
            "chrf" : "chrF"}

height_dict = {"syll_dist" : 450,
            "syll_acc" : 500,
            "rhyme_scheme_agree" : 450,
            "rhyme_accuracy" : 500,
            "semantic_sim" : 100,
            "keyword_sim" : 100,
            "line_keyword_sim" : 120,
            "phon_rep_dif" : 180,
            "bleu4gram" : 700,
            "bleu2gram" : 700,
            "chrf" : 350}




with open("target_results_dict.json", "w", encoding="utf-8") as json_file:
    json.dump(results_dict, json_file, ensure_ascii=False)

for cat in results_dict:
    print(f"{cat} -> {sum(results_dict[cat]) / len(results_dict[cat])}")

    if cat == "syll_dist":
        bins = np.linspace(0, 3, 91)
    else:
        bins = np.linspace(0, 1, 31)

    plt.hist(results_dict[cat], bins=bins, density=False, alpha=0.75, color='b')
    plt.xlabel('Values', fontsize=20)
    plt.ylabel('Frequencies', fontsize=20)   
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.ylim(0, height_dict[cat])
    
    # Display the plot
    if RANDOM_BASELINE == True:
        plt.title(f"Baseline - {name_dict[cat]}", fontsize=20)
        plt.savefig(f"random_baseline_{cat}.pdf", format="pdf", bbox_inches="tight")
    elif HT == True:
        plt.title(f"Target - {name_dict[cat]}", fontsize=20)
        plt.savefig(f"HT_{cat}.pdf", format="pdf", bbox_inches="tight")
    else:
        plt.title(f"Lindat - {name_dict[cat]}", fontsize=20)
        plt.savefig(f"sth_{cat}.pdf", format="pdf", bbox_inches="tight")
    
    plt.show()


# RANDOM
# syll_dist -> 0.647128599750164
# syll_acc -> 0.09944603419179703
# rhyme_scheme_agree -> 0.5544133832269426
# rhyme_accuracy -> 0.19881190971329638
# semantic_sim -> 0.2313272277257751
# keyword_sim -> 0.34168958166565383
# line_keyword_sim -> 0.21761956677711425
# phon_rep_dif -> 0.12407562340733677
# bleu4gram -> 5.01326109678458e-157
# bleu2gram -> 0.0009352034288523701
# chrf -> 0.0903275489758511

# syll_dist -> 0.6731137266099138
# syll_acc -> 0.087752380952381
# rhyme_scheme_agree -> 0.5840333333333335
# rhyme_accuracy -> 0.24178231768231753
# semantic_sim -> 0.16725276211276652
# keyword_sim -> 0.2519295328259468
# line_keyword_sim -> 0.17096886224605684
# phon_rep_dif -> 0.12406592435423497
# bleu2gram -> 0.00022142462456595065
# chrf -> 0.08875831496193098

# TRUE
# syll_dist -> 0.031397304724096894
# syll_acc -> 0.8262192139310786
# rhyme_scheme_agree -> 0.7669619928094502
# rhyme_accuracy -> 0.598514642351314
# semantic_sim -> 0.616103348475759
# keyword_sim -> 0.6424479474914827
# line_keyword_sim -> 0.4489510799626657
# phon_rep_dif -> 0.08314464719285819
# bleu4gram -> 0.003080892565414774
# bleu2gram -> 0.03889312066055365
# chrf -> 0.16327499136429904



# espeak
# scheme_agreement_to_cs -> 0.7669619928094502
# scheme_agreement_to_en -> 0.8274359820969988

# ipa transcribers
# scheme_agreement_to_cs -> 0.6820884388680993
# scheme_agreement_to_en -> 0.8425232959131264


##########################
# Correct thingy Aligned #
##########################

# syll_dist -> 0.031397304724096894
# syll_acc -> 0.8262192139310786
# rhyme_scheme_agree -> 0.6379081370606793 SWRT
# semantic_sim -> 0.6004684052856292
# keyword_sim -> 0.49629239412813875
# line_keyword_sim -> 0.5003293137231976
# phon_rep_dif -> 0.08314464719285819
# bleu -> 0.03911828525782059       2gram
# bleu -> 0.0037377977622871255     4gram
# chrf -> 0.1670763720679358

#########################
# Correct thingy Random #
#########################

# syll_dist -> 0.664016752610145
# syll_acc -> 0.10378237581627422
# rhyme_scheme_agree -> 0.5207150194438329
# semantic_sim -> 0.2142703992979047
# keyword_sim -> 0.13133081679127662
# line_keyword_sim -> 0.2606570512207164
# phon_rep_dif -> 0.12695510999946247
# bleu -> 0.0007215634926773624
# chrf -> 0.09278634330189597






# True alignment - new

# syll_dist -> 0.031397304724096894
# syll_acc -> 0.8262192139310786
# rhyme_scheme_agree -> 0.6379081370606793 SWRT
# semantic_sim -> 0.6004684052856292
# keyword_sim -> 0.49629239412813875
# line_keyword_sim -> 0.5003293137231976
# phon_rep_dif -> 0.0661192632056612
# bleu -> 0.0
        
# random
        
# syll_dist -> 0.6560583994295848
# syll_acc -> 0.08977181011079323
# rhyme_scheme_agree -> 0.5440164355418593 SWRT
# semantic_sim -> 0.20998964708109022
# keyword_sim -> 0.13740818885310002
# line_keyword_sim -> 0.2658511946233171
# phon_rep_dif -> 0.11356770014443346
# bleu -> 0.0















# True alignment:

    # syll_dist -> 0.03589051233543377
    # syll_acc -> 0.8322604729176896
    # rhyme_scheme_agree -> 0.6314852315496646    rhymetagger
    # semantic_sim -> 0.5961625609047634
    # keyword_sim -> 0.5273546302287849
    # line_keyword_sim -> 0.5178069276220453
        

    # syll_dist -> 0.03589051233543377
    # syll_acc -> 0.8322604729176896
    # rhyme_scheme_agree -> 0.30460944589488403    rhyme finder
    # semantic_sim -> 0.5961625609047634
    # keyword_sim -> 0.5273546302287849
    # line_keyword_sim -> 0.5178069276220453
    

    # syll_dist -> 0.03589051233543377
    # syll_acc -> 0.8322604729176896
    # rhyme_scheme_agree -> 0.7582404798094232     same word rhymetagger  
    # semantic_sim -> 0.5961625609047634
    # keyword_sim -> 0.5273546302287849
    # line_keyword_sim -> 0.5178069276220453


# Random Baseline:
    
    # 1

    # syll_dist -> 0.822100400261972
    # syll_acc -> 0.09610970641944099
    # rhyme_scheme_agree -> 0.6329273774406518      same word rhymetagger  
    # semantic_sim -> 0.2120982622365113
    # keyword_sim -> 0.31691310407709233
    # line_keyword_sim -> 0.28426752490771295

    # 2

    # syll_dist -> 0.8413033861471998
    # syll_acc -> 0.09222994802640827
    # rhyme_scheme_agree -> 0.6349703393951186      same word rhymetagger  
    # semantic_sim -> 0.21603904031876442
    # keyword_sim -> 0.31473833603240486
    # line_keyword_sim -> 0.28169275645958103


    # syll_dist -> 0.8305076866852645
    # syll_acc -> 0.09328627616238241
    # rhyme_scheme_agree -> 0.587109144542773       rhymetagger
    # semantic_sim -> 0.21781719207961475
    # keyword_sim -> 0.3160961678399976
    # line_keyword_sim -> 0.2816791897849417