import os
import json
from statistics import mean, median
from syllabator import syllabify

VZ_PATH = "./DATA/Velky_zpevnik"
HT_PATH = "./DATA"

with open(os.path.join(VZ_PATH, 'VZ.json'), "r", encoding="utf-8") as json_file:
    VZ_dataset = json.load(json_file)
with open(os.path.join(HT_PATH, 'HT.json'), "r", encoding="utf-8") as json_file:
    HT_dataset = json.load(json_file)


VZ_endings = [VZ_dataset[x]["line_endings"] for x in VZ_dataset]
VZ_tagger = [VZ_dataset[x]["rhymes"] for x in VZ_dataset]

HT_endings = []
HT_tagger = []
for musical_name in HT_dataset["cs"]:  
    if musical_name == "num_sections":
            continue
    for song_name in HT_dataset["cs"][musical_name]:
        if song_name == "num_sections":
            continue
        for sec_i in HT_dataset["cs"][musical_name][song_name]:
            HT_endings.append(HT_dataset["cs"][musical_name][song_name][sec_i]["line_endings"])
            HT_tagger.append(HT_dataset["cs"][musical_name][song_name][sec_i]["rhymes"])
            
endings_list = VZ_endings + HT_endings
tagger_list = VZ_tagger + HT_tagger

common_rhymes_dict = {}

# endings_list = [["a", "a", "b", "c"], ["c", "a", "b", "c"]]

# find all groupings
for section_id in range(len(endings_list)):
    sec_endings = endings_list[section_id]
    sec_tags = tagger_list[section_id]

    for i in range(len(sec_endings)):
        sec_end = sec_endings[i]

        if sec_end not in common_rhymes_dict:
            common_rhymes_dict[sec_end] = {"count" : 0, "rhymes" : {}}
        common_rhymes_dict[sec_end]["count"] += 1

        for j in range(len(sec_endings)):
            if i == j:
                continue

            rhyme_end = sec_endings[j]

            koef = 1
            if sec_tags[i] == sec_tags[j]:
                koef = 1

            if rhyme_end not in common_rhymes_dict[sec_end]["rhymes"]:
                common_rhymes_dict[sec_end]["rhymes"][rhyme_end] = 0
            common_rhymes_dict[sec_end]["rhymes"][rhyme_end] += koef

# remove unfrequent endings

# compute probabilities that each rhyme is grouped exactly with that ending
for sec_end in common_rhymes_dict:
    for rhyme_end in common_rhymes_dict[sec_end]["rhymes"]:
        common_rhymes_dict[sec_end]["rhymes"][rhyme_end] = (common_rhymes_dict[sec_end]["rhymes"][rhyme_end] * 2) / (common_rhymes_dict[rhyme_end]["count"] + common_rhymes_dict[sec_end]["count"])


# sort rhymes 
for sec_end in common_rhymes_dict:
    sorted_rhymes = sorted(common_rhymes_dict[sec_end]["rhymes"].items(), key=lambda x:x[1], reverse=True) 
    common_rhymes_dict[sec_end]["rhymes"] = dict(sorted_rhymes)

# sort
sorted_dict = sorted(common_rhymes_dict.items(), key=lambda x:x[1]["count"], reverse=False)
common_rhymes_dict = dict(sorted_dict)

rhymes_counts = [common_rhymes_dict[x]["count"] for x in common_rhymes_dict]
print("avg endings count", mean(rhymes_counts))
print("median endings count", median(rhymes_counts))
s_r_c = sorted(rhymes_counts)
print(len(s_r_c))
print(s_r_c[len(s_r_c)-500])



# print x most probable for all
for sec_end in common_rhymes_dict:
    if common_rhymes_dict[sec_end]["count"] < 124:
        continue
    counter = 0
    print(sec_end, common_rhymes_dict[sec_end]["count"], "\n-------------")
    for rhyme_end in common_rhymes_dict[sec_end]["rhymes"]:
        print(rhyme_end, common_rhymes_dict[sec_end]["rhymes"][rhyme_end], common_rhymes_dict[rhyme_end]["count"])
        counter += 1
        if counter > 20:
            break
    print("---------------")



with open(f"common_rhymes.json", "a", encoding='utf-8') as json_file:
    json.dump(common_rhymes_dict, json_file, ensure_ascii=False)




# napasovat na rymova schemata a EM? mozna?