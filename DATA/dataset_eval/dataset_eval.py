import random
import os
import json
from statistics import mean, median

VZ_PERCENTAGE = 0.001
HT_PERCENTAGE = 0.02

VZ_PATH = "./"
HT_PATH = "./"

with open(os.path.join(VZ_PATH, 'VZ.json'), "r", encoding="utf-8") as json_file:
    VZ_dataset = json.load(json_file)
with open(os.path.join(HT_PATH, 'HT.json'), "r", encoding="utf-8") as json_file:
    HT_dataset = json.load(json_file)

VZ_ids = random.sample(range(1, len(VZ_dataset)), int(len(VZ_dataset) * VZ_PERCENTAGE))
HT_ids = random.sample(range(1, HT_dataset["cs"]["num_sections"]), int(HT_dataset["cs"]["num_sections"] * HT_PERCENTAGE))

VZ_samples = [VZ_dataset[str(x)]["lyrics"] for x in VZ_ids]

HT_list = []
for musical_name in HT_dataset["cs"]:  
    if musical_name == "num_sections":
            continue
    for song_name in HT_dataset["cs"][musical_name]:
        if song_name == "num_sections":
            continue
        for sec_i in HT_dataset["cs"][musical_name][song_name]:
            HT_list.append(HT_dataset["cs"][musical_name][song_name][sec_i]["lyrics"])
HT_samples = [HT_list[x] for x in HT_ids]

samples = VZ_samples + HT_samples

samples_ids = list(range(len(samples)))
random.shuffle(samples_ids)

VZ_results = [0] * len(VZ_samples)
HT_results = [0] * len(HT_samples)

NAME = input("Jmeno: ")
print()

print("Ohodnotte nasledujici sekce textu podle toho jak moc si myslite, ze by sekce mohla byt sloka pisne:\n\n1 absolutni jistota ze to takhle ma byt, tu melodii v tom temer slysim\n2 dokazu si predstavit ze tohle asi sloka bude, ale nevim to 100%\n3 jo, dokazu akceptovat ze tohle je sloka. Ale mam pocit ze by mela jeste pokracovat/neco by ji melo predchazet\n4 jakoze jako sloka to nezni, ale je to usek textu ve spravnem formatu\n5 je tam bordel a metadata, typu text: Jan Novak, akordy: A Gmi F\n\nPro predcasne ukonceni hodnoceni misto score napiste 'end'\n\nPro spusteni stisknete enter")
print()
print("="*30)
input()

counter = 1
for i in samples_ids:
    print(f"{counter}/{len(samples_ids)}")
    print()
    for line in samples[i]:
        print(line)
    print()

    score = input("Score: ")
    while score not in {"1","2","3","4","5","end"}:
        print("Score musi byt mezi 1(nejlepsi) a 5(nejhorsi)")
        score = input("Score: ")

    counter+=1
    
    if score == "end":
        break

    if i < len(VZ_samples):
        VZ_results[i] = int(score)
    else:
        HT_results[i - len(VZ_samples)] = int(score)
    
    print()
    print("="*30)
    print()

VZ_non_zeros = [x for x in VZ_results if x > 0]
HT_non_zeros = [x for x in HT_results if x > 0]

results_dict = {"VZ" : {"samples" : VZ_samples, "results" : VZ_results, "avg" : mean(VZ_non_zeros), "median" : median(VZ_non_zeros)},
                "HT" : {"samples" : HT_samples, "results" : HT_results, "avg" : mean(HT_non_zeros), "median" : median(HT_non_zeros)}}

with open(f"results_{NAME}.json", "a", encoding='utf-8') as json_file:
    json.dump(results_dict, json_file, ensure_ascii=False)

print("Sections evaluated: ", counter - 1)
print("Average dataset score: ", results_dict["VZ"]["avg"])
print("Median dataset score: ", results_dict["VZ"]["median"])
print("Average musicals score: ", results_dict["HT"]["avg"])
print("Median musicals score: ", results_dict["HT"]["median"])

input()
