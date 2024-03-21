from HT_loader import HT_loader
from evaluator import Evaluator
from english_structure_extractor import SectionStructure


evaluator = Evaluator()

cs_lyrics = HT_loader("DATA", "cs")
en_lyrics = HT_loader("DATA", "en")

assert len(en_lyrics) == len(cs_lyrics)

HT_pairs = []
for i in range(len(en_lyrics)):
    HT_pairs.append((cs_lyrics[i], SectionStructure(section=en_lyrics[i], kw_model=evaluator.kw_model, rt=evaluator.rt_cs)))
    
results_dict = evaluator.evaluate_outputs_structure(HT_pairs)

for cat in results_dict:
    print(f"{cat} -> {sum(results_dict[cat]) / len(results_dict[cat])}")