import random
from HT_loader import HT_loader
import json
import os

def create(outName):
    en_lyrics = HT_loader("./", "en")
    cs_lyrics = ["\n".join(lyric.split(",")) for lyric in HT_loader("./", "cs")]

    with open("CODE/Human_eval/HumEVresults_dicts\\OSCAR_GPT2_whole_dataset_type_5_epoch_0_samples_649_out_per_generation_10_stopwords_True_rhymer_3_fewshot_10.json", "r", encoding="utf-8") as json_file:
        dataset_dict = json.load(json_file)
    ogpt2_wsk = ["\n".join(lyric[1]) for lyric in dataset_dict["lyrics"]]

    with open("CODE/Human_eval/HumEVresults_dicts\\VUT_GPT2_whole_dataset_type_5_epoch_0_samples_649_out_per_generation_10_stopwords_True_rhymer_3_fewshot_10.json", "r", encoding="utf-8") as json_file:
        dataset_dict = json.load(json_file)
    bgpt2_wsk = ["\n".join(lyric[1]) for lyric in dataset_dict["lyrics"]]

    with open("CODE/Human_eval/HumEVresults_dicts\\TINYLLAMA_lines_dataset_type_5_epoch_0_samples_649_out_per_generation_10_stopwords_True_rhymer_3_fewshot_10.json", "r", encoding="utf-8") as json_file:
        dataset_dict = json.load(json_file)
    tl_lsk = ["\n".join(lyric[1]) for lyric in dataset_dict["lyrics"]]

    with open("CODE/Human_eval/HumEVresults_dicts\\VUT_TINYLLAMA_lines_dataset_type_7_epoch_0_samples_649_out_per_generation_10_stopwords_True_rhymer_3_fewshot_10.json", "r", encoding="utf-8") as json_file:
        dataset_dict = json.load(json_file)
    btl_lst = ["\n".join(lyric[1]) for lyric in dataset_dict["lyrics"]]

    with open("CODE/Human_eval/HumEVresults_dicts\\VUT_TINYLLAMA_whole_dataset_type_13_epoch_0_samples_649_out_per_generation_10_stopwords_True_rhymer_3_fewshot_10.json", "r", encoding="utf-8") as json_file:
        dataset_dict = json.load(json_file)
    btl_wskr = ["\n".join(lyric[1]) for lyric in dataset_dict["lyrics"]]

    with open("czech_machine_translations_list.json", "r", encoding="utf-8") as json_file:
        dataset_dict = json.load(json_file)
    translation = ["\n".join(lyric.split(",")) for lyric in dataset_dict]

    with open("CODE\Human_eval\HumEVresults_dicts\mistral10Y.json", "r", encoding="utf-8") as json_file:
        dataset_dict = json.load(json_file)
    mistral = ["\n".join(lyric) for lyric in dataset_dict]

    model_outs_to_evaluate = {0:ogpt2_wsk, 1:tl_lsk, 3:btl_wskr, 5:btl_lst, 6:bgpt2_wsk, 2:cs_lyrics, 4:translation, 7:mistral}

    output_text = []


    output_text.append(f"# Ceka vas 28 uryvku originalnich pisnicek a jejich dvou moznych cover verzi (~15 min)\n#\n#\n# Urcete lepsi cover!\n# Prioritne:\n#     1. je mozne ho zazpivat na stejnou melodii jako original (neznate-li melodii nejakou si vymyslete) \n#     2. je napsany prirozene (slova na sebe navazuji) \n#     3. uryvek zni jako slova k pisni (zni zpevne, ne 100% smysluplne nebo verne originalu)\n#\n#\n# Jak na to?\n#    - prectete si original a obe moznosti\n#    - !!Vzdy!! zvolte lepsi cover (1 nebo 2)\n#    - Napiste cislo favorita do kolonky \"Lepsi :\"\n#    ... jako v nasledujici ukazce\n#\n#\n#  !!!! KROME KOLONKY LEPSI NIC NEPREPISUJTE, NEMENTE !!!!!\n#\n# Priklad spravneho vyplneni:\n#")

    output_text.append(50 * "#")
    output_text.append(f"# Lyrics 0 (ukazka)\n#")
    en_og = "\n".join(["# Mama take this badge from me", "# I can't use it anymore", "# It's getting dark too dark to see", "# Feels like I'm knockin' on Heaven's door", "# Knock-knock-knockin' on Heaven's door", "# Knock-knock-knockin' on Heaven's door"])
    output_text.append(f"# Original:\n#\n{en_og}\n#\n#")

    output_text.append("# 1.)")
    cover1 = "\n".join(["# Mámo sundej kytku z kabátu", "# Za chvíli nebude k ničemu", "# Můj kabát a kudlu vem pro tátu", "# Já už stojím před bránou s klíčem", "# Cejtím že zaklepu na nebeskou bránu", "# Cejtím že zaklepu na nebeskou bránu"])
    output_text.append(f"{cover1}\n#\n#")

    output_text.append("# 2.)")
    cove2 = "\n".join(["# mámo vem si věci mý", "# já už vím, že půjdu tmou", "# zkus ztratit slovo, slovo jediný", "# a na nebi dvůr jistě mi otevřou", "# Na, na, na nebi otevřou", "# Na, na, na nebi otevřou"])
    output_text.append(f"{cove2}\n#\n#")

    output_text.append(f"# Lepsi: 2\n#")

    output_text.append(50 * "#")

    counter = 1
    for i in range(len(model_outs_to_evaluate)):
        for j in range(i+1, len(model_outs_to_evaluate)):

            section_id = random.randint(0, len(model_outs_to_evaluate[0]) - 1)

            output_text.append(f"Lyrics {counter}/28\n")
            counter += 1

            en_og = "\n".join(en_lyrics[section_id].split(","))
            output_text.append(f"Original:\n\n{en_og}\n\n")

            output_text.append("1.)")
            output_text.append(f"{model_outs_to_evaluate[i][section_id]}\n\n")

            output_text.append("2.)")
            output_text.append(f"{model_outs_to_evaluate[j][section_id]}\n\n")

            output_text.append(f"Lepsi: \n")
            
            output_text.append(30 * "=")


    with open(os.path.join("CODE/Human_eval/generated_files/", outName), "w", encoding="utf-8") as file:
        file.write("\n".join(output_text))





for key in range(4,5):
    create(f"human_eval_covers_{key}.txt")