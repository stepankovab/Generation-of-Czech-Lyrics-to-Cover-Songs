from HT_loader import HT_loader
from evaluator import Evaluator
from english_structure_extractor import SectionStructure
from eval.rhyme_finder import RhymeFinder
from eval.tagger import RhymeTagger
import matplotlib.pyplot as plt


evaluator = Evaluator(rt=RhymeTagger())

cs_lyrics = HT_loader("DATA", "cs")
en_lyrics = HT_loader("DATA", "en")

assert len(en_lyrics) == len(cs_lyrics)

HT_pairs = []
for i in range(len(en_lyrics)):
    HT_pairs.append((cs_lyrics[i], SectionStructure(section=en_lyrics[i], kw_model=evaluator.kw_model, rt=evaluator.rt)))
    
results_dict = evaluator.evaluate_outputs_structure(HT_pairs)

for cat in results_dict:
    print(f"{cat} -> {sum(results_dict[cat]) / len(results_dict[cat])}")

    plt.hist(results_dict[cat], bins=30, color="blue")
    
    # Adding labels and title
    plt.xlabel('Values')
    plt.ylabel('Frequency')
    plt.title(cat)
    
    # Display the plot
    plt.savefig(f"both_finder_{cat}.png")


# syll_dist -> 0.03589051233543377
# syll_acc -> 0.8322604729176896
# rhyme_scheme_agree -> 0.6314852315496646    rhymetagger
# semantic_sim -> 0.5961625609047634
# keyword_sim -> 0.5273546302287849
# line_keyword_sim -> 0.5178069276220453