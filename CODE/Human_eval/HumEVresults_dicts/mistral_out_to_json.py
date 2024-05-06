import json




with open("CODE\LLMExperiments\HumEVresults_dicts\mistral10Y.txt", "r", encoding="utf-8") as file:
    lines = file.readlines()


outputs = []

section = []
breaks_counter = 0
for line in lines:
    if "===" in line:
        outputs.append(section[:-1])
        section.clear()
        breaks_counter = 0

    if breaks_counter >= 2:
        section.append(line[:-1])

    if not line.strip():
        breaks_counter += 1 


with open("CODE\LLMExperiments\HumEVresults_dicts\mistral10Y.json", "w", encoding="utf-8") as json_file:
    json.dump(outputs, json_file, ensure_ascii=False)


    






