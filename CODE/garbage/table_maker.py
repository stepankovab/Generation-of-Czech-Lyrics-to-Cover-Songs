import csv
import numpy as np
from matplotlib import pyplot as plt

model2num = {"oscar" : 0,
             "vutGpt" : 1,
             "vutLlama" : 2,
             "Llama" : 3,
             "mist5": 4,
             "mist10": 5}

prompt2num = {"w1" : 0,
              "w3" : 1,
              "w5" : 2,
              "w13" : 3,
              "l1" : 4,
              "l3" : 5,
              "l5" : 6,
              "l7" : 7}

name_dict = {6 : "Syllable Distance",
            7 : "Syllable Accuracy",
            8 : "Rhyme Scheme Agreement",
            9 : "Rhyme Scheme Accuracy",
            10 : "Semantic Similarity",
            11 : "Keyword Similarity",
            12 : "Line-by-line Keyword Similarity",
            13 : "Phoneme Repetition Difference",
            14 : "BLEU (2-gram)",
            15 : "chrF"}

model_name_dict = {"oscar" : "Czech-GPT2-OSCAR",
             "vutGpt" : "Czech-GPT-2-XL-133k",
             "vutLlama" : "CSTinyLlama-1.2B",
             "Llama" : "TinyLlama-1.1B"}


def get_one_metric_modelXprompt():
    for EVAL_CAT in range(6,16):

        # syll_dist	syll_acc	rhyme_scheme_agree	rhyme_accuracy	semantic_sim	keyword_sim	line_keyword_sim	phon_rep_dif	bleu2gram	chrf
        for_graph = [[0 for _ in range(8)] for _ in range(6)]

        with open('model_eval_sheet.csv', newline='') as csvfile:
            spamreader = csv.reader(csvfile, delimiter=',', quotechar='|')
            for row in spamreader:
                if row[5] == "LINDAT":
                    translator = [round(float(row[EVAL_CAT]),2) for _ in range(8)]
                if row[5] == "HUMAN":
                    target = [round(float(row[EVAL_CAT]),2) for _ in range(8)]
                if row[5] == "RANDOM":
                    random = [round(float(row[EVAL_CAT]),2) for _ in range(8)]
                if row[1] in model2num and row[2] in prompt2num and row[3] == "0" and row[4] == "1":
                    for_graph[model2num[row[1]]][prompt2num[row[2]]] = round(float(row[EVAL_CAT]),2)

        fig, ax = plt.subplots(frameon=False)

        rows = ["OGPT2", "BGPT2", "BTL", "TL"]
        columns = ["WS", "WK", "WSK", "WSKR", "LS", "LK", "LSK", "LST"]

        rows = ["Random baseline"] + rows + ["MIS-5shot", "MIS-10shot"] + ["Target"]
        for_graph = [random] + for_graph + [target] # + [[0.4 for i in range(4)] + ["-" for i in range(4)], [0.5 for i in range(4)] + ["-" for i in range(4)]]


        color_data = np.array(for_graph)
        # normed_data = (conf_data - conf_data.min(axis=0, keepdims=True)) / conf_data.ptp(axis=0, keepdims=True)

        for i in range(4,8):
            for_graph[5][i] = "-"
            for_graph[6][i] = "-"
        conf_data = np.array(for_graph)

        fig.patch.set_visible(False)
        ax.axis('off')
        ax.axis('tight')

        if EVAL_CAT == 6 or EVAL_CAT == 13:
            table = ax.table(cellText=conf_data,
                        rowLabels=rows,
                        colLabels=columns,
                        cellColours=plt.cm.viridis_r(color_data),
                        loc='center',
                        colWidths=[0.1 for x in columns]
                        )
        else:
            table = ax.table(cellText=conf_data,
                        rowLabels=rows,
                        colLabels=columns,
                        cellColours=plt.cm.viridis(color_data),
                        loc='center',
                        colWidths=[0.1 for x in columns]
                        )

        for cell in table._cells:
            table._cells[cell].set_alpha(.7)

        cellDict = table.get_celld()
        for i in range(4,8):
            cellDict[(7, i)].set_color("white")
            cellDict[(6, i)].set_color("white")
            cellDict[(7, i)].set_edgecolor("black")
            cellDict[(6, i)].set_edgecolor("black")


        # table.set_fontsize(150)

        fig.tight_layout()
        # plt.show()

        
        fig.canvas.draw()
        bbox = table.get_window_extent(fig.canvas.get_renderer())
        bbox_inches = bbox.transformed(fig.dpi_scale_trans.inverted())
        plt.savefig(f"pure_out_table_{'_'.join(name_dict[EVAL_CAT].split())}.pdf", format="pdf", bbox_inches=bbox_inches)

        plt.close()

def get_epochs_graph(model_name, prompt_type):

    graph_data = [[0 for _ in range(4)] for _ in range(10)]

    with open('model_eval_sheet.csv', newline='') as csvfile:
        spamreader = csv.reader(csvfile, delimiter=',', quotechar='|')
        for row in spamreader:
            if row[1] == model_name and row[2] == prompt_type and row[4] == "1" and row[3] in "0123":
                for met_i in range(10):
                    graph_data[met_i][int(row[3])] = round(float(row[met_i + 6]),2)

    prompt_name = prompt_type
    if prompt_type == "w5":
        prompt_name = "Whole - Syllables and keywords"
    if prompt_type == "l7":
        prompt_name = "Lines - Syllables and translations"

    for i in range(10):
        plt.plot(list(range(1,5)), graph_data[i], label = name_dict[6 + i])
    plt.xlabel('Epoch')
    plt.ylabel('Value')
    plt.title(f"{model_name_dict[model_name]} - {prompt_name}")
    if model_name == "oscar":
        plt.legend(bbox_to_anchor =(0.0,-0.75), loc='lower left')
    plt.xticks([1,2,3,4])
    plt.ylim(0,1)
    plt.savefig(f"epochs_{model_name_dict[model_name]}_{prompt_type}.pdf", format="pdf", bbox_inches="tight")

    plt.close()

def get_one_metric_bestmodelXpostproc():
    NUM_COL = 6

    for EVAL_CAT in range(6,16):

        # syll_dist	syll_acc	rhyme_scheme_agree	rhyme_accuracy	semantic_sim	keyword_sim	line_keyword_sim	phon_rep_dif	bleu2gram	chrf
        for_graph = [[] for _ in range(6)]
        translator = [[] for _ in range(NUM_COL)]
        random = [[] for _ in range(NUM_COL)]
        target = [[] for _ in range(NUM_COL)]

        with open('model_eval_sheet.csv', newline='') as csvfile:
            spamreader = csv.reader(csvfile, delimiter=',', quotechar='|')
            for row in spamreader:
                if row[5] == "LINDAT":
                    translator = [round(float(row[EVAL_CAT]),2) for _ in range(NUM_COL)]
                if row[5] == "HUMAN":
                    target = [round(float(row[EVAL_CAT]),2) for _ in range(NUM_COL)]
                if row[5] == "RANDOM":
                    random = [round(float(row[EVAL_CAT]),2) for _ in range(NUM_COL)]
                
                if row[3] == "N":
                    if row[4] == "1":
                        for_graph[0].append(round(float(row[EVAL_CAT]),2))
                    if row[4] == "5":
                        for_graph[1].append(round(float(row[EVAL_CAT]),2))
                    if row[4] == "10":
                        for_graph[2].append(round(float(row[EVAL_CAT]),2))
                if row[3] == "Y":
                    if row[4] == "1":
                        for_graph[3].append(round(float(row[EVAL_CAT]),2))
                    if row[4] == "5":
                        for_graph[4].append(round(float(row[EVAL_CAT]),2))
                    if row[4] == "10":
                        for_graph[5].append(round(float(row[EVAL_CAT]),2))

        
        fig, ax = plt.subplots(frameon=False)

        rows = ["Raw output", "Choose from 5", "Choose from 10", "Stopwords", "Stopwords + Choose from 5", "Stopwords + Choose from 10"]
        columns = ["BTL\nWSKR", "BGPT2\nWSK", "OGPT2\nWSK", "TL\nLSK", "BTL\nLST", "F-MIS\nWSKR"]

        rows = ["Random baseline"] + rows + ["Target"]
        for_graph = [random] + for_graph + [target]

        conf_data = np.array(for_graph)

        
        # normed_data = (conf_data - conf_data.min(axis=0, keepdims=True)) / conf_data.ptp(axis=0, keepdims=True)

        fig.patch.set_visible(False)
        ax.axis('off')
        ax.axis('tight')

        if EVAL_CAT == 6 or EVAL_CAT == 13:
            table = ax.table(cellText=conf_data,
                        rowLabels=rows,
                        colLabels=columns,
                        cellColours=plt.cm.viridis_r(conf_data),
                        loc='center',
                        colWidths=[0.1 for x in columns]
                        )
        else:
            table = ax.table(cellText=conf_data,
                        rowLabels=rows,
                        colLabels=columns,
                        cellColours=plt.cm.viridis(conf_data),
                        loc='center',
                        colWidths=[0.1 for x in columns]
                        )

        for cell in table._cells:
            table._cells[cell].set_alpha(.7)

        cellDict = table.get_celld()
        for i in range(0,len(columns)):
            cellDict[(0,i)].set_height(.08)

        table.set_fontsize(30)

        

        fig.tight_layout()
        # plt.show()
        
        
        fig.canvas.draw()
        bbox = table.get_window_extent(fig.canvas.get_renderer())
        bbox_inches = bbox.transformed(fig.dpi_scale_trans.inverted())
        plt.savefig(f"postprocess_table_{'_'.join(name_dict[EVAL_CAT].split())}.pdf", format="pdf", bbox_inches=bbox_inches)

        plt.close()



get_one_metric_bestmodelXpostproc()
# get_one_metric_modelXprompt()


















# for model in model_name_dict.keys():
#     get_epochs_graph(model, "w5")