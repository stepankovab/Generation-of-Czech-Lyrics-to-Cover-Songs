import os
import re
from HT_loader import HT_loader


with open(os.path.join("CODE","LLMExperiments","outs_eval", "outputs.o21441454"), mode="r", encoding="utf-8") as f:
    lines = f.readlines()

cs_lyrics = []
for i in range(len(lines)):
    line = lines[i]
    if re.match("after postprocessing", line):
        cs_lyrics.append(",".join(lines[i+2].strip().split(", ")))

en_lyrics = HT_loader("./", "en")





