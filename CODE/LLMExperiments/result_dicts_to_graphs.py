import os
import json
import re
import matplotlib.pyplot as plt



filenames = []
for filename in os.listdir("results_dicts"):
    if re.match(r"OSCAR_GPT2", filename):
        filenames.append(filename)

for filename in filenames:
    with open(os.path.join("results_dicts", filename), "r", encoding="utf-8") as json_file:
        results_dict = json.load(json_file)


    for cat in results_dict:
        if cat == "lyrics":
            continue
        print(f"{cat} -> {sum(results_dict[cat]) / len(results_dict[cat])}")

        plt.hist(results_dict[cat], bins=30, color="blue", range=(0,1))
        
        # Adding labels and title
        plt.xlabel('Values')
        plt.ylabel('Frequency')
        plt.title(cat)
        
        # Display the plot
        plt.savefig(f"x_{cat}.png")