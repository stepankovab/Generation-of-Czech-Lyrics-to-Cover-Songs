from HT_loader import HT_loader
from evaluator import Evaluator
from english_structure_extractor import SectionStructure
from eval.rhyme_finder import RhymeFinder
from eval.tagger import RhymeTagger
from eval.same_word_tagger import SameWordRhymeTagger
from eval.syllabator import syllabify as cs_syll
from eval.en_syllabator import syllabify as en_syll
from rhymer_types import RhymerType
import matplotlib.pyplot as plt
import random
import numpy as np

RANDOM_BASELINE = True

evaluator = Evaluator(czech_rhyme_detector=RhymerType.SAME_WORD_RHYMETAGGER)
structure = SectionStructure(kw_model=evaluator.kw_model, english_rhyme_detector=RhymerType.SAME_WORD_RHYMETAGGER)
# rt_cs=SameWordRhymeTagger("cs")
# rt_en=SameWordRhymeTagger("en")

cs_lyrics = HT_loader("./", "cs")
en_lyrics = HT_loader("./", "en")

if RANDOM_BASELINE:
    random.shuffle(en_lyrics)

assert len(en_lyrics) == len(cs_lyrics)

HT_pairs = []
# results_dict = {"syll_dist" : []
#                 }

for i in range(len(en_lyrics)):
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
        en_split = en_lyrics[i].split(',')
        shorter_len = min(len(cs_split), len(en_split))
        cs_lyrics[i] = ','.join(cs_split[:shorter_len])
        en_lyrics[i] = ','.join(en_split[:shorter_len])

        cs_split = cs_split[:shorter_len]
        en_split = en_split[:shorter_len]

    
    # en_sylls = []
    # for i in range(len(en_split)):
    #     en_sylls.append(len(en_syll(en_split[i])))
    
    # cs_sylls = []
    # for i in range(len(cs_split)):
    #     cs_sylls.append(len(cs_syll(cs_split[i])))

    # results_dict["syll_dist"].append(evaluator.get_section_syllable_distance(cs_sylls, en_sylls))

    structure.fill(en_lyrics[i])
    HT_pairs.append((cs_lyrics[i], structure.copy()))
    
results_dict = evaluator.evaluate_outputs_structure(HT_pairs, evaluate_keywords=True, evaluate_line_keywords=True, evaluate_translations=True)

name_dict = {"syll_dist" : "Syllable Distance",
            "syll_acc" : "Syllable Accuracy",
            "rhyme_scheme_agree" : "Rhyme Scheme Agreement",
            "rhyme_accuracy" : "Rhyme Scheme Accuracy",
            "semantic_sim" : "Semantic Similarity",
            "keyword_sim" : "Keyword Similarity",
            "line_keyword_sim" : "Line-by-line Keywords Similarity",
            "phon_rep_dif" : "Phoneme Repetition Difference",
            "bleu4gram" : "BLEU (4-gram)",
            "bleu2gram" : "BLEU (2-gram)",
            "chrf" : "chrF"}

for cat in results_dict:
    print(f"{cat} -> {sum(results_dict[cat]) / len(results_dict[cat])}")

    bins = np.linspace(0, 3, 91)
    plt.hist(results_dict[cat], bins=bins, density=False, alpha=0.75, color='b')
    plt.xlabel('Values')
    plt.ylabel('Frequency')   
    
    # Display the plot
    if RANDOM_BASELINE == True:
        plt.title(f"Random baseline - {name_dict[cat]}")
        plt.savefig(f"random_baseline_{cat}.png")
    else:
        plt.title(f"Human Translations - {name_dict[cat]}")
        plt.savefig(f"HT_{cat}.png")
    
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