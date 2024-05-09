import os
from matplotlib import pyplot as plt
import numpy as np

def evalUate_HE(filename, winner = ""):
    model_outs_to_evaluate = {0:"ogpt2_wsk", 1:"tl_lsk", 3:"btl_wskr", 5:"btl_lst", 6:"bgpt2_wsk", 2:"cs_lyrics", 4:"translation", 7:"mistral"}

    with open(filename, "r", encoding="utf-8") as file:
        output_text = file.readlines()

    order = []
    for i in range(len(model_outs_to_evaluate)):
        for j in range(i+1, len(model_outs_to_evaluate)):
            order.append((i,j))

    ratings = []
    for line in output_text:
        if "Lepsi:" in line:
            for c in line:
                try:
                    ratings.append(int(c))
                    break
                except:
                    pass

    ratings = ratings[1:]

    assert len(ratings) == len(order)

    winner = {"ogpt2_wsk":0, "tl_lsk":0, "btl_wskr":0, "btl_lst":0, "bgpt2_wsk":0, "cs_lyrics":0, "translation":0,"mistral":0}
    for i in range(len(order)):
        winner[model_outs_to_evaluate[order[i][ratings[i] - 1]]] += 1
        
    return winner

MODEL = "btl_wskr"

at_least = {"cs_lyrics":[], "translation":[],"btl_wskr":[],"bgpt2_wsk":[],"ogpt2_wsk":[],"btl_lst":[],"tl_lsk":[],"mistral":[]}
more_than = {"cs_lyrics":[], "translation":[],"btl_wskr":[],"bgpt2_wsk":[],"ogpt2_wsk":[],"btl_lst":[],"tl_lsk":[],"mistral":[]}
at_least_total = []
more_than_total = []
for i in range(8):
    at_least_total.append(0)
    more_than_total.append(0)
    for key in at_least:
        at_least[key].append(0)
        more_than[key].append(0)
    for file in os.listdir("CODE\\Human_eval\\evaluated files"):
        a = evalUate_HE(os.path.join("CODE\\Human_eval\\evaluated files",file))

        if a[MODEL] <= i:
            pass
        else:
            at_least_total[-1] += 1
            for k in a:
                at_least[k][-1] += a[k]

        if a[MODEL] > i:
            pass
        else:
            more_than_total[-1] += 1
            for k in a:
                more_than[k][-1] += a[k]

    for key in at_least:
        if at_least_total[-1] > 0:
            at_least[key][-1] = round((at_least[key][-1] / at_least_total[-1]) / 7, 2)
        if more_than_total[-1] > 0:
            more_than[key][-1] = round((more_than[key][-1] / more_than_total[-1]) / 7, 2)
        
    


#     print(f"Evaluated by {total} people:")

#     sorted_winner = sorted(at_least, key=lambda x: at_least[x], reverse=True)
#     # if sorted_winner[0] == "translation":
#     #     pass
#     # else:
#     for x in sorted_winner:
#         print(f"{x} {round(at_least[x] / total, 2)}/{7} = {round(at_least[x]/ (total * 7),2)}")



name_dict = {"cs_lyrics": "Human-written song translations", "translation": "Machine translations", "btl_wskr":"BUT TinyLlama - WSKR", "bgpt2_wsk" : "BUT GPT2 - WSK"}


plt.plot([x for x in range(8)][:-1], [x / max(at_least_total) for x in at_least_total][:-1], label = "participants", color='black', linewidth = 5, alpha=0.2)
for key in at_least:
    plt.plot([x for x in range(8)][:-1], at_least[key][:-1], label = key)
plt.xlabel(f"minimum points given to {MODEL}", fontsize=18) # more
# plt.ylabel('percentage')
plt.xticks([x for x in range(8)][:-1], fontsize=18)
plt.yticks(fontsize=18)
plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.65), fontsize=18, ncol=2)
plt.ylim(0,1)
plt.savefig(os.path.join("CODE\\Human_eval", f"at_least_{MODEL}.pdf"), format="pdf", bbox_inches="tight")

plt.close()

plt.plot([x for x in range(8)][1:], [x / max(more_than_total) for x in more_than_total][1:], label = "participants", color='black', linewidth = 5, alpha=0.2)
for key in more_than:
    plt.plot([x for x in range(8)][1:], more_than[key][1:], label = key)
plt.xlabel(f"maximum points given to {MODEL}", fontsize=18) # less than or equal to
plt.ylabel('percentage', fontsize=18)
plt.xticks([x for x in range(8)][1:], fontsize=18)
plt.yticks(fontsize=18)
plt.ylim(0,1)
plt.title(name_dict[MODEL], fontsize=25, loc='right', y=1.3)
plt.savefig(os.path.join("CODE\\Human_eval", f"more_than_{MODEL}.pdf"), format="pdf", bbox_inches="tight")

plt.close()

