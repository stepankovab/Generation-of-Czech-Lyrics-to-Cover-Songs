# from eval.tagger import RhymeTagger
# from HT_loader import HT_loader


# rt = RhymeTagger()

# rt.load_model("cs")

# cs_data = HT_loader("./", "cs")

# for lyrics in cs_data:
#     print(lyrics)
#     lyrics = lyrics.split(",")
#     scheme = rt.tag(lyrics, output_format=3)
#     print(scheme)
#     print()

# import re

# print(re.sub(r'\bsedm(náct|desát|krát|\b)', r'sedu\1', "ahoj sedmikrátsko"))

import json
from evaluator import Evaluator

with open("english_HT_rhymes.json", "r", encoding="utf-8") as json_file:
    my_own = json.load(json_file)
with open("english_HT_rhymes_espeak.json", "r", encoding="utf-8") as json_file:
    espeak = json.load(json_file)

assert len(my_own) == len(espeak)

ev = Evaluator()

accs = []
esp2mys = []
my2esps = []
for i in range(len(my_own)):
    acc = ev.get_rhyme_scheme_accuracy(espeak[i], my_own[i])
    esp2my = ev.get_rhyme_scheme_agreement(espeak[i], my_own[i])
    my2esp = ev.get_rhyme_scheme_agreement(my_own[i], espeak[i])
    accs.append(acc)
    esp2mys.append(esp2my)
    my2esps.append(my2esp)
    print("acc", acc)
    print("esp2my", esp2my)
    print("my2esp", my2esp)
    print()

print("avg acc  -> ", sum(accs)/len(accs))
print("avg esp2my  -> ", sum(esp2mys)/len(esp2mys))
print("avg my2esp  -> ", sum(my2esps)/len(my2esps))


# cs
# avg acc  ->  0.7810447085870813
# avg esp2my  ->  0.8194756279502041
# avg my2esp  ->  0.96226795803067

# en
# avg acc  ->  0.9055264941107414
# avg esp2my  ->  0.9330838652872553
# avg my2esp  ->  0.972325555800132