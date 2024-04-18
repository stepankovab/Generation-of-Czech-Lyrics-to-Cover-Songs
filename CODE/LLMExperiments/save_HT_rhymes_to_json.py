from HT_loader import HT_loader
from eval.same_word_tagger import SameWordRhymeTagger
# from evaluator import Evaluator
import json

RANDOM_BASELINE = False

tagger = SameWordRhymeTagger("cs")
en_lyrics = HT_loader("./", language="cs")

rhymes = []
for i in range(len(en_lyrics)):
    rhymes.append(tagger.tag(en_lyrics[i].split(',')))

print("==================== dumping structures into json =====================")
with open("czech_HT_rhymes.json", "w", encoding='utf-8') as json_file:
    json.dump(rhymes, json_file, ensure_ascii=False)


