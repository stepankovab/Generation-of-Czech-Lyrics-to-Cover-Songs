from HT_loader import HT_loader
from evaluator import Evaluator
from english_structure_extractor import SectionStructure
# from eval.rhyme_finder import RhymeFinder
# from eval.tagger import RhymeTagger
from eval.same_word_tagger import SameWordRhymeTagger
import matplotlib.pyplot as plt
import random


RANDOM_BASELINE = True

evaluator = Evaluator(rt=SameWordRhymeTagger())

cs_lyrics = HT_loader("./", "cs")
en_lyrics = HT_loader("./", "en")

if RANDOM_BASELINE:
    random.shuffle(en_lyrics)

assert len(en_lyrics) == len(cs_lyrics)

HT_pairs = []
for i in range(len(en_lyrics)):
    if RANDOM_BASELINE:
        cs_split = cs_lyrics[i].split(',')
        en_split = en_lyrics[i].split(',')
        shorter_len = min(len(cs_split), len(en_split))
        cs_lyrics[i] = ','.join(cs_split[:shorter_len])
        en_lyrics[i] = ','.join(en_split[:shorter_len])

    HT_pairs.append((cs_lyrics[i], SectionStructure(section=en_lyrics[i], kw_model=evaluator.kw_model, rt=evaluator.rt)))
    
results_dict = evaluator.evaluate_outputs_structure(HT_pairs, evaluate_keywords=True, evaluate_line_keywords=True, evaluate_translations=True)

for cat in results_dict:
    print(f"{cat} -> {sum(results_dict[cat]) / len(results_dict[cat])}")

    plt.hist(results_dict[cat], bins=30, color="blue", range=(0,1))
    
    # Adding labels and title
    plt.xlabel('Values')
    plt.ylabel('Frequency')
    plt.title(cat)
    
    # Display the plot
    if RANDOM_BASELINE == True:
        plt.savefig(f"random_baseline_{cat}.png")
    else:
        plt.savefig(f"HT_{cat}.png")


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
# bleu -> 0.03911828525782059
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