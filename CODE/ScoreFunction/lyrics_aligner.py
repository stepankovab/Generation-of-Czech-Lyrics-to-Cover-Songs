import re
from syllabator import syllabify
from phoneme_unifier import english_to_czech_pron
import requests
import numpy as np
import math
import os


def dynamically_align_lyrics_by_section_and_line(czech_text : str, english_text : str, line_range : int) -> list[list[str]]:
    '''
    Aligns lyrics to minimize deviation.

    `line_range` specifies number of surrounding lines to consider when aligning.

    returns aligned lyrics, fist czech then english
    '''
    # czech_lines = [a for a in re.split(r"\s*[,\.\n:!?;]+\s*", re.sub('\n\n', '\n<>\n', czech_text)) if re.search(r'(\w+)|(<>)', a)]
    # english_lines = [a for a in re.split(r"\s*[,\.\n:!?;]+\s*", re.sub('\n\n', '\n<>\n', english_text)) if re.search(r'(\w+)|(<>)', a)]

    print(len(syllabify(czech_text, "cz")))
    print(len(syllabify(english_text, "en")))

    czech_lines = [a for a in re.split(r"\s*[,\.\n:!?;]+\s*", czech_text) if re.search(r'(\w+)|(<>)', a)] + ["<eos>"]
    english_lines = [a for a in re.split(r"\s*[,\.\n:!?;]+\s*", english_text) if re.search(r'(\w+)|(<>)', a)] + ["<eos>"]

    dp = [[0 for x in range(len(english_lines))] for x in range(len(czech_lines))]

    for i_cz in range(len(czech_lines)):
        for i_en in range(len(english_lines)):
            if abs(i_cz - i_en) > abs(len(czech_lines) - len(english_lines)) + line_range * 2:
                dp[i_cz][i_en] = math.inf
                continue

            czech_line_len = len(syllabify(czech_lines[i_cz], "cz"))
            english_line_len = len(syllabify(english_lines[i_en], "en"))
 
            if i_cz == 0 and i_en == 0:
                dp[i_cz][i_en] = czech_line_len - english_line_len

            elif i_cz == 0:
                dp[i_cz][i_en] = dp[i_cz][i_en - 1] - english_line_len

            elif i_en == 0:
                dp[i_cz][i_en] = dp[i_cz - 1][i_en] + czech_line_len

            elif czech_line_len == english_line_len:
                dp[i_cz][i_en] = dp[i_cz - 1][i_en - 1]

            else:
                possibilities = [dp[i_cz-1][i_en-1] + czech_line_len - english_line_len,
                                 dp[i_cz][i_en - 1] - english_line_len,
                                 dp[i_cz - 1][i_en] + czech_line_len]
                absolute_argmin = np.argmin([abs(x) for x in possibilities])
                dp[i_cz][i_en] = possibilities[absolute_argmin]
                
    ## return dp

    min_diff = dp[len(czech_lines) - 1][len(english_lines) - 1]

    i_en = len(english_lines) - 1
    i_cz = len(czech_lines) - 1

    center_points = []

    while i_cz >= 0 and i_en >= 0:
        min_diff = dp[i_cz][i_en]

        if abs(len(syllabify(czech_lines[i_cz], "cz")) - len(syllabify(english_lines[i_en], "en"))) <= 1:
            center_points.append((i_cz, i_en))

        possibilities = [(i_cz - x, i_en - y) for x in reversed(range(line_range)) for y in reversed(range(line_range)) if (x,y) != (0,0)]
        absolute_argmin = np.argmin([abs(abs(dp[x_cz][x_en]) - abs(min_diff)) for x_cz, x_en in possibilities])

        i_cz = possibilities[absolute_argmin][0]
        i_en = possibilities[absolute_argmin][1]

    if center_points[0] != (len(czech_lines) - 1, len(english_lines) - 1):
        center_points = [(len(czech_lines) - 1, len(english_lines) - 1)] + center_points
    
    if center_points[-1] != (0,0):
        center_points.append((0,0))

    c = 0
    while c < len(center_points) - 1:
        if center_points[c + 1][0] - center_points[c][0] == 0 or center_points[c + 1][1] - center_points[c][1] == 0:
            center_points.pop(c)
            continue
        c += 1

    aligned_czech_lines = []
    aligned_english_lines = []

    for c in range(len(center_points) - 1):
        c_cz = center_points[c][0]
        c_en = center_points[c][1]

        center_point_diff = dp[c_cz][c_en]
        small_dp = np.array(dp)[center_points[c + 1][0]:c_cz, center_points[c + 1][1]:c_en]
        small_dp -= center_point_diff
        small_dp = abs(small_dp)

        prev_small_en = math.inf

        for small_cz in reversed(range(len(small_dp))):
            small_en = np.argmin(small_dp[small_cz])
            if small_dp[small_cz][small_en] <= 1 and (prev_small_en == small_en + 1 or prev_small_en == math.inf):
                aligned_czech_lines.append(czech_lines[center_points[c + 1][0] + small_cz])
                aligned_english_lines.append(english_lines[center_points[c + 1][1] + small_en])

                prev_small_en = small_en
                continue

            if prev_small_en == small_en:
                aligned_czech_lines[-1] = (czech_lines[center_points[c + 1][0] + small_cz]) + " <> " + aligned_czech_lines[-1]
                continue

            if prev_small_en >= small_en + 1:
                current_en = prev_small_en - 1
                while small_en < current_en:
                    aligned_english_lines[-1] = (english_lines[center_points[c + 1][1] + current_en]) + " <> " + aligned_english_lines[-1]
                    current_en -= 1

            if small_dp[small_cz][small_en] <= 1:
                aligned_czech_lines.append(czech_lines[center_points[c + 1][0] + small_cz])
                aligned_english_lines.append(english_lines[center_points[c + 1][1] + small_en])

                prev_small_en = small_en

            else:
                aligned_czech_lines[-1] = (czech_lines[center_points[c + 1][0] + small_cz]) + " <> " + aligned_czech_lines[-1]
                aligned_english_lines[-1] = (english_lines[center_points[c + 1][1] + small_en]) + " <> " + aligned_english_lines[-1]

                prev_small_en = small_en

    if len(aligned_czech_lines) != len(aligned_english_lines):
        raise ValueError("The lyrics differ in number of lines.")
    
    n_lines = len(aligned_czech_lines)

    final_czech_lines = []
    final_english_lines = []

    for n in reversed(range(n_lines)):
        cz_line = aligned_czech_lines[n]
        en_line = aligned_english_lines[n]

        split_cz_line = cz_line.split("<>")
        split_en_line = en_line.split("<>")

        if len(split_en_line) == len(split_cz_line) == 1:
            final_czech_lines.append(cz_line)
            final_english_lines.append(en_line)
            continue
        
        split_cz_line_syl = [syllabify(x, "cz") for x in split_cz_line]
        split_en_line_syl = [syllabify(x, "en") for x in split_en_line]

        cz_syl_sum = [0]
        for syl_line in split_cz_line_syl:
            cz_syl_sum.append(cz_syl_sum[-1] + len(syl_line))
        cz_syl_sum = cz_syl_sum[1:]

        en_syl_sum = [0]
        for syl_line in split_en_line_syl:
            en_syl_sum.append(en_syl_sum[-1] + len(syl_line))
        en_syl_sum = en_syl_sum[1:]

        # TODO do properly

        final_czech_lines.append(" ".join(split_cz_line))
        final_english_lines.append(" ".join(split_en_line))

    return final_czech_lines, final_english_lines


# dynamically_align_lyrics_by_section_and_line("Mám styl Čendy, Nevím kde ho Čenda vzal, Já ho sebral od něho, A on si ho moc nehlídal",
#                             "I'm still standing, better than I've ever been, Feeling like a true survivor, feeling like a little kid", 3)

# with open("C:/Users/barca/MOJE/ROCNIKAC/rp-barbora-stepankova/DATA/Musicals/" + "moana_02" + "_cs.txt", "r", encoding="utf-8") as f:
#     text_cs = f.read()

# with open("C:/Users/barca/MOJE/ROCNIKAC/rp-barbora-stepankova/DATA/Musicals/" + "moana_02" + "_en.txt", "r", encoding="utf-8") as f:
#     text_en = f.read()


# czech_lines, english_lines = dynamically_align_lyrics_by_section_and_line(text_cs, text_en, 3)

# for i in range(len(czech_lines)):
#     print(czech_lines[i], english_lines[i])




def align_lyrics_by_section_and_line(czech_text : str, english_text : str) -> list[list[str]]:
    '''
    Aligns lyrics

    returns aligned lyrics, fist czech then english
    '''
    czech_lines = [a for a in re.split(r"\s*[\.()\n:!?;]+\s*", re.sub('\n\n', '\n<eos>\n', re.sub(',', '', czech_text))) if re.search('\w+', a)]
    english_lines = [a for a in re.split(r"\s*[()\.\n:!?;]+\s*", re.sub('\n\n', '\n<eos>\n', english_text)) if re.search('\w+', a)]

    czech_final = []
    english_final = []

    czech_temp = []
    english_temp = []

    i_cz = 0
    i_en = 0

    deviation = 0

    while i_cz + 1 < len(czech_lines) and i_en + 1 < len(english_lines):
        if czech_lines[i_cz] == "<eos>" or english_lines[i_en] == "<eos>":
            czech_final.append(czech_temp)
            english_final.append(english_temp)
            czech_temp = []
            english_temp = []
            
            while czech_lines[i_cz] == "<eos>":
                i_cz += 1

            while english_lines[i_en] == "<eos>":
                i_en += 1

            if i_cz + 1 >= len(czech_lines) or i_en + 1 >= len(english_lines):
                break

        if czech_lines[i_cz][0] == '#' or english_lines[i_en][0] == '#':
            czech_temp.append(czech_lines[i_cz][1:])
            english_temp.append(english_lines[i_en][1:])
            i_cz += 1
            i_en += 1
            continue

        cz_syl = syllabify(czech_lines[i_cz], "cz")
        en_line_pronounciation = english_to_czech_pron(english_lines[i_en])
        en_syl = syllabify(english_lines[i_en], "en")

        # to be quick
        if len(cz_syl) == len(en_syl):
            czech_temp.append(czech_lines[i_cz])
            english_temp.append(english_lines[i_en])
            i_cz += 1
            i_en += 1
            continue

        # if the line lengths don't match up exactly, choose the closest version.
        next_cz_syl = syllabify(czech_lines[i_cz + 1], "cz")
        next_en_syl = syllabify(english_lines[i_en + 1], "en")

        one_cz_one_en_dif = abs(len(cz_syl) - len(en_syl))
        one_cz_two_en_dif = abs(len(cz_syl) - len(next_en_syl) - len(en_syl))
        two_cz_one_en_dif = abs(len(cz_syl) + len(next_cz_syl) - len(en_syl))
        two_cz_two_en_dif = abs(len(cz_syl) + len(next_cz_syl) - len(next_en_syl) - len(en_syl))

        min_distance = min(one_cz_one_en_dif, one_cz_two_en_dif, two_cz_one_en_dif, two_cz_two_en_dif)

        # TODO better mistake-catching system!!!
        if min_distance > 2:
            pass

        deviation += min_distance

        # the lines are actually quite close together syllabically
        if min_distance == one_cz_one_en_dif or czech_lines[i_cz + 1] == "<eos>" or english_lines[i_en + 1] == "<eos>":
            czech_temp.append(czech_lines[i_cz])
            english_temp.append(english_lines[i_en])
            i_cz += 1
            i_en += 1

        # one czech line matches two english lines
        elif min_distance == one_cz_two_en_dif:
            # czech line can be split into two by english standard
            if cz_syl[len(en_syl) - 1] + " " in czech_lines[i_cz]:
                stop_index = czech_lines[i_cz].index(cz_syl[len(en_syl) - 1]) + len(cz_syl[len(en_syl) - 1]) 

                czech_temp.append(czech_lines[i_cz][:stop_index])
                czech_temp.append(czech_lines[i_cz][stop_index + 1:])
                i_cz += 1

                english_temp.append(english_lines[i_en])
                i_en += 1
                english_temp.append(english_lines[i_en])
                i_en += 1
                
            # czech line cant be split, merge english
            else:
                english_temp.append(english_lines[i_en] + " " + english_lines[i_en + 1])
                i_en += 2

                czech_temp.append(czech_lines[i_cz])
                i_cz += 1

        # one english line matches two czech lines
        elif min_distance == two_cz_one_en_dif:
            # english line can be split into two by czech standard
            if en_syl[len(cz_syl) - 1] + " " in en_line_pronounciation:
                stop_index = en_line_pronounciation.index(en_syl[len(cz_syl) - 1]) + len(en_syl[len(cz_syl) - 1]) 
                words_on_first_line = len(en_line_pronounciation[:stop_index].split())

                czech_temp.append(czech_lines[i_cz])
                i_cz += 1
                czech_temp.append(czech_lines[i_cz])
                i_cz += 1

                english_temp.append(" ".join(english_lines[i_en].split()[:words_on_first_line]))
                english_temp.append(" ".join(english_lines[i_en].split()[words_on_first_line:]))
                i_en += 1

            # english line cant be split, merge czech
            else:
                czech_temp.append(czech_lines[i_cz] + " " + czech_lines[i_cz + 1])
                i_cz += 2

                english_temp.append(english_lines[i_en])
                i_en += 1
        
        # merge two next lines together in both
        else:
            czech_temp.append(czech_lines[i_cz] + " " + czech_lines[i_cz + 1])
            english_temp.append(english_lines[i_en] + " " + english_lines[i_en + 1])
            i_cz += 2
            i_en += 2

    # after there is just one line left in the shorter lyrics
    czech_final_line = ""
    for i in range(i_cz, len(czech_lines)):
        if czech_lines[i] == "<eos>":
            continue
        czech_final_line += czech_lines[i] + ", "
    english_final_line = ""
    for j in range(i_en, len(english_lines)):
        if english_lines[j] == "<eos>":
            continue
        english_final_line += english_lines[j] + ", "
    
    czech_temp.append(czech_final_line[:-2])
    english_temp.append(english_final_line[:-2])

    czech_final.append(czech_temp)
    english_final.append(english_temp)

    return czech_final, english_final



# with open("C:/Users/barca/MOJE/ROCNIKAC/rp-barbora-stepankova/DATA/Musicals/" + "moana_01" + "_cs.txt", "r", encoding="utf-8") as f:
#     text_cs = f.read()

# with open("C:/Users/barca/MOJE/ROCNIKAC/rp-barbora-stepankova/DATA/Musicals/" + "moana_01" + "_en.txt", "r", encoding="utf-8") as f:
#     text_en = f.read()

# dp = dynamically_align_lyrics_by_section_and_line(text_cs, text_en, 4)


# with open("C:/Users/barca/MOJE/ROCNIKAC/rp-barbora-stepankova/DATA/Musicals/" + "moana_02" + "_cs.txt", "r", encoding="utf-8") as f:
#     text_cs = f.read()

# with open("C:/Users/barca/MOJE/ROCNIKAC/rp-barbora-stepankova/DATA/Musicals/" + "moana_02" + "_en.txt", "r", encoding="utf-8") as f:
#     text_en = f.read()

# c, e = align_lyrics_by_section_and_line(text_cs, text_en)

# czech_lines = []
# for section in c:
#     czech_lines.extend(section)

# english_lines = []
# for section in e:
#     english_lines.extend(section)

# for i in range(len(czech_lines)):
#     print(czech_lines[i], english_lines[i])

# print(5)




def get_aligned_translation(sections, target_language):
    '''
    target_language = ['cz', 'en']
    '''
    lines = []
    for section in sections:
        lines.extend(section)

    joined = "\n".join(lines)

    if target_language == "en":
        url = 'http://lindat.mff.cuni.cz/services/translation/api/v2/models/cs-en'
    elif target_language == "cz":
        url = 'http://lindat.mff.cuni.cz/services/translation/api/v2/models/en-cs'
    response = requests.post(url, data = {"input_text": joined})
    response.encoding='utf8'
    translated_joined = response.text

    translated = translated_joined.split("\n")[:-1]

    translated_sections = []

    for section in sections:
        translated_sections.append(translated[:len(section)])
        translated = translated[len(section):]

    return translated_sections


# title = "encanto_01"

# with open("C:/Users/barca/MOJE/UMEA/NLP-Umea/FinalProject/DATA/Human_translations/" + title + "_cs.txt", "r", encoding="utf-8") as f:
#     text_cs = f.read()

# with open("C:/Users/barca/MOJE/UMEA/NLP-Umea/FinalProject/DATA/Human_translations/" + title + "_en.txt", "r", encoding="utf-8") as f:
#     text_en = f.read()


# czech_sections, english_sections = align_lyrics_by_section_and_line(text_cs, text_en)

# for section_i in range(len(czech_sections)):
#     for line_i in range(len(czech_sections[section_i])):
#         diff = len(syllabify(czech_sections[section_i][line_i], "cz")) - len(syllabify(english_sections[section_i][line_i], "en"))
#         print(czech_sections[section_i][line_i], " --- ", english_sections[section_i][line_i], "--->", diff)
#     print()


# if input() == "Y":
#     with open("C:/Users/barca/MOJE/UMEA/NLP-Umea/FinalProject/DATA/Aligned_HT/" + title + "_cs.txt", "w", encoding="utf-8") as cz_doc:
#         for section_i in range(len(czech_sections)):
#             for line_i in range(len(czech_sections[section_i])):
#                 cz_doc.write(czech_sections[section_i][line_i] + "\n")
#             cz_doc.write("\n")
#     with open("C:/Users/barca/MOJE/UMEA/NLP-Umea/FinalProject/DATA/Aligned_HT/" + title + "_en.txt", "w", encoding="utf-8") as en_doc:
#         for section_i in range(len(english_sections)):
#             for line_i in range(len(english_sections[section_i])):
#                 en_doc.write(english_sections[section_i][line_i] + "\n")
#             en_doc.write("\n")
