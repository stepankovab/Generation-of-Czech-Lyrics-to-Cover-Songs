import os
import json
import numpy

def HT_loader(dataset_path, language = "en", without_repetition = False):
    lyrics_path = os.path.join(dataset_path, 'HT.json')

    with open(lyrics_path, "r", encoding="utf-8") as json_file:
        dataset_dict = json.load(json_file)

    if language == "en" or language == "cs":
        HT_lyrics = _extract_lyrics(dataset_dict, language)

    elif language == "both":
        HT_lyrics = _extract_lyrics(dataset_dict, "cs") + _extract_lyrics(dataset_dict, "en")

    else:
        HT_lyrics = []

    if without_repetition:
        HT_lyrics = list(set(HT_lyrics))

    return HT_lyrics

def _extract_lyrics(dataset_dict, language):
    list_of_sections = []

    for mov in dataset_dict[language]:
        if mov == "num_sections":
            continue
        for song in dataset_dict[language][mov]:
            if song == "num_sections":
                continue
            for section in dataset_dict[language][mov][song]:
                list_of_sections.append(','.join(dataset_dict[language][mov][song][section]["lyrics"]))

    return list_of_sections





def avg_line_len(language):

    with open("DATA/HT.json", "r", encoding="utf-8") as json_file:
        dataset_dict = json.load(json_file)

    line_lens_dict = {}
    values = []

    c_a = 0
    c_b = 0
    c_c = 0

    for mov in dataset_dict[language]:
        if mov == "num_sections":
            continue
        c_a += 1
        for song in dataset_dict[language][mov]:
            if song == "num_sections":
                continue
            c_b += 1
            for section in dataset_dict[language][mov][song]:
                c_c += 1
                for line in dataset_dict[language][mov][song][section]["lyrics"]:
                    if len(line) not in line_lens_dict:
                        line_lens_dict[len(line)] = 0
                    line_lens_dict[len(line)] += 1
                    values.append(len(line))

    x = numpy.quantile(values, [0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1])
    print(x)

    with open("DATA/Velky_zpevnik/VZ.json", "r", encoding="utf-8") as json_file:
        dataset_dict = json.load(json_file)

    line_lens_dict = {}
    values = []

    for section in dataset_dict:
        for line in dataset_dict[section]["lyrics"]:
            if len(line) not in line_lens_dict:
                line_lens_dict[len(line)] = 0
            line_lens_dict[len(line)] += 1
            values.append(len(line))

    x = numpy.quantile(values, [0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1])
    print(x)

# avg_line_len("cs")


# [ 0  .1  .2  .3  .4  .5  .6  .7  .8  .9  1]
#
# [ 5. 17. 21. 24. 28. 31. 34. 37. 41. 47. 1218.]
# [ 2.  9. 12. 15. 17. 20. 22. 25. 28. 34. 84.]
# [ 4. 10. 14. 17. 19. 21. 24. 27. 31. 36. 55.]