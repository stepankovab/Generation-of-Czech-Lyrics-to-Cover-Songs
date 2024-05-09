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

    # for i in range(len(order)):
    #     winner[order[i][ratings[i] - 1]][order[i][1 - (ratings[i] - 1)]][0] += 1
    #     winner[order[i][1 - (ratings[i] - 1)]][order[i][ratings[i] - 1]][1] += 1

    return winner

total = 0
winner = {"cs_lyrics":0, "translation":0, "btl_wskr":0, "ogpt2_wsk":0, "tl_lsk":0,"btl_lst":0, "bgpt2_wsk":0, "mistral":0}
for file in os.listdir("CODE\\Human_eval\\evaluated files"):
    a = evalUate_HE(os.path.join("CODE\\Human_eval\\evaluated files",file))
    if a["translation"] <= 5:  # not (a["btl_wskr"] >= 4 and a["bgpt2_wsk"] >= 3 and a["translation"] <= 5):
        total += 1
        for k in a:
            winner[k] += a[k]
# total = len(os.listdir('CODE/Human_eval/evaluated files'))

# for file in os.listdir("CODE\\Human_eval\\evaluated files"):
#     print()
#     print(file)

# winner = evalUate_HE(os.path.join("CODE\\Human_eval\\evaluated files", "manual_eval_covers_14.txt"))
# total = 1

# model_outs_to_evaluate = {0:"ogpt2_wsk", 1:"tl_lsk", 3:"btl_wskr", 5:"btl_lst", 6:"bgpt2_wsk", 2:"cs_lyrics", 4:"translation", 7:"mistral"}
# winner = [[[0,0] for _ in range(len(model_outs_to_evaluate))] for _ in range(len(model_outs_to_evaluate))]
# for file in os.listdir("CODE\\Human_eval\\evaluated files"):
#     winner = evalUate_HE(os.path.join("CODE\\Human_eval\\evaluated files",file), winner)

print(f"Evaluated by {total} people: -> {total/len(os.listdir('CODE/Human_eval/evaluated files'))}")

sorted_winner = sorted(winner, key=lambda x: winner[x], reverse=True)
# if sorted_winner[0] == "translation":
#     pass
# else:
for x in sorted_winner:
    print(f"{x} {round(winner[x] / total, 2)}/{7} = {round(winner[x]/ (total * 7),2)}")


# fig, ax = plt.subplots(frameon=False)
# rows = [model_outs_to_evaluate[x] for x in range(len(model_outs_to_evaluate))]
# cols = ["\n".join(model_outs_to_evaluate[x].split("_")) for x in range(len(model_outs_to_evaluate))]
# for_graph = [[f"{winner[j][i][0]}:{winner[j][i][1]}" for i in range(len(model_outs_to_evaluate))] for j in range(len(model_outs_to_evaluate))]

# conf_data = np.array(for_graph)


# # normed_data = (conf_data - conf_data.min(axis=0, keepdims=True)) / conf_data.ptp(axis=0, keepdims=True)

# fig.patch.set_visible(False)
# ax.axis('off')
# ax.axis('tight')

# table = ax.table(cellText=conf_data,
#             rowLabels=rows,
#             colLabels=cols,
#             loc='center',
#             colWidths=[0.1 for x in cols]
#                 )

# for cell in table._cells:
#     table._cells[cell].set_alpha(.7)

# cellDict = table.get_celld()
# for i in range(0,len(rows)):
#     cellDict[(0,i)].set_height(.08)

# # table.set_fontsize(30)



# fig.tight_layout()
# # plt.show()


# fig.canvas.draw()
# bbox = table.get_window_extent(fig.canvas.get_renderer())
# bbox_inches = bbox.transformed(fig.dpi_scale_trans.inverted())
# plt.savefig("CODE/Human_eval/results.pdf", format="pdf", bbox_inches=bbox_inches)

# plt.close()