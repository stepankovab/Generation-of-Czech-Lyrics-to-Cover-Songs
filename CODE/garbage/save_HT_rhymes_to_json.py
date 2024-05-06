from HT_loader import HT_loader
# from english_structure_extractor import SectionStructureExtractor
import json
from translate import lindat_translate

en_lyrics = HT_loader("./", language="en")
# extractor = SectionStructureExtractor()

# structure_list = []
# for i in range(len(en_lyrics)):
#     print(i)
#     structure = extractor.create_section_structure(en_lyrics[i])
#     structure_list.append(structure.to_dict())

translations_list = []

for i in range(len(en_lyrics)):
    split_en = en_lyrics[i].split(",")
    cs_entered = lindat_translate(split_en, "en", "cs", "\n")
    split_cs = cs_entered.replace(',', '').split("\n")
    if len(split_cs) != len(split_en):
        en_entered = '\n'.join(split_en)
        print(f"{i}\n\n{en_entered}\n\n{cs_entered}\n\n")
        translations_list.append("")
        continue
    for line_i in range(len(split_cs)):
        split_cs[line_i] = split_cs[line_i].strip(" .,:\'\"!?")

    translations_list.append(','.join(split_cs))


print("==================== dumping structures into json =====================")
with open("czech_machine_translations_list.json", "w", encoding='utf-8') as json_file:
    json.dump(translations_list, json_file, ensure_ascii=False)


