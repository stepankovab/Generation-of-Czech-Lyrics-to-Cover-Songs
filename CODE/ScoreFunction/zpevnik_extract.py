import re
import math
from langdetect import detect_langs
from phoneme_repetition_similarity import phoneme_distinct2


MAX_LINES = 8
MIN_LINES = 3

MAX_LINE_LEN = 55
TRY_SPLIT_LINE_LEN = 25
MIN_LINE_LEN = 5

def detect_language_with_langdetect(line): 
    try: 
        langs = detect_langs(line) 
        for item in langs: 
            # The first one returned is usually the one that has the highest probability
            return item.lang, item.prob 
    except: return "err", 0.0 

def split_section_based_on_distinct2(temp_section : list[str], min_lines : int = 2) -> tuple[list[str]]:
    """
    Splits the list of lines on the best splitting point considering distinct2 score.

    Parameters
    -----------
    temp_section: The list of lines to be split
    min_lines: Lower bound on the line count per section

    Returns
    ------
    Tupple containing the split sections
    """
    best_split_value = math.inf
    best_split_i = 0

    for i in range(min_lines, len(temp_section) - min_lines + 1):
        phon2_1 = phoneme_distinct2(temp_section[:i], "cz")
        phon2_2 = phoneme_distinct2(temp_section[i:], "cz")

        if phon2_1 + phon2_2 < best_split_value: # and -> so its rozumny sekce
            best_split_value = phon2_1 + phon2_2
            best_split_i = i

    return (temp_section[:best_split_i], temp_section[best_split_i:])

def recursive_section_split(sum_lines, temp_section : list[str], counter : int, max_lines : int = 10, min_lines : int = 2) -> int:
    """
    Recursively splits a grouped verse that is longer then the upper bound on the line count.
    
    Parameters
    -----------
    temp_section: The list of lines to be split
    counter: Number of the section saved into file
    max_lines: Upper bound on the line count per section
    min_lines: Lower bound on the line count per section
    
    Returns
    ---------
    Counter after saving sections
    """
    (temp_section_1, temp_section_2) = split_section_based_on_distinct2(temp_section, min_lines)

    if len(temp_section_1) > max_lines:
        (sum_lines, counter) = recursive_section_split(sum_lines, temp_section_1, counter, max_lines)
    else:
        with open("DATA\\Velky_zpevnik\\VZ_sections\\zpevnik_" + str(counter) + ".txt", "w", encoding="utf-8") as new_section:
            new_section.writelines(temp_section_1)
        counter += 1
        sum_lines += len(temp_section_1)

    if len(temp_section_2) > max_lines:
        (sum_lines, counter) = recursive_section_split(sum_lines, temp_section_2, counter, max_lines)
    else:
        with open("DATA\\Velky_zpevnik\\VZ_sections\\zpevnik_" + str(counter) + ".txt", "w", encoding="utf-8") as new_section:
            new_section.writelines(temp_section_2)
        counter += 1
        sum_lines += len(temp_section_2)

    return sum_lines, counter


def section_join(sum_lines, short_section : list[str], next_section : list[str], counter : int, max_lines : int = 10, min_lines : int = 2) -> tuple[int, str, bool] :
    return_short = []
    return_short_bool = False

    with open("DATA\\Velky_zpevnik\\VZ_sections\\zpevnik_" + str(counter - 1) + ".txt", "r", encoding="utf-8") as prev_section_file:
        prev_section = prev_section_file.readlines()

    sum_lines -= len(prev_section)

    phon2_prev = phoneme_distinct2(prev_section + short_section, "cz")
    phon2_next = phoneme_distinct2(short_section + next_section, "cz")

    if phon2_prev < phon2_next:
        first_section = prev_section + short_section
        second_section = next_section
    else:
        first_section = prev_section
        second_section = short_section + next_section

    if len(first_section) > max_lines:
        sum_lines, counter = recursive_section_split(sum_lines, first_section, counter - 1, max_lines, min_lines)
    else:
        with open("DATA\\Velky_zpevnik\\VZ_sections\\zpevnik_" + str(counter - 1) + ".txt", "w", encoding="utf-8") as new_section:
            new_section.writelines(first_section)
        sum_lines += len(first_section)
    
    if len(second_section) > max_lines:
        sum_lines, counter = recursive_section_split(sum_lines, second_section, counter, max_lines, min_lines)
    elif len(second_section) < min_lines:
        return_short = second_section
        return_short_bool = True
    else:
        with open("DATA\\Velky_zpevnik\\VZ_sections\\zpevnik_" + str(counter) + ".txt", "w", encoding="utf-8") as new_section:
            new_section.writelines(second_section)
        counter += 1
        sum_lines += len(second_section)

    return (sum_lines, counter, return_short, return_short_bool)






with open("DATA\\Velky_zpevnik\\velkyzpevnik.txt", "r", encoding="utf-8") as file:
    lines = file.readlines()


deletions=[
    r"\bR\b",
    r"\br\b",
    r"\brf\b",
    r"\bRef\b",
    r"\bREF\b",
    r"\bref\b",
    r"\bRefrén\b",
    r"\bRefrain\b",
    r"\b[CDEFGHBXxcdefghb][bs#]*\b",
    r"\b[CDEFGHBXxcdefghb]mi\b",
    r"\b[CDEFGHBXxcdefghb][éá]\b",
    r"[0-9]+"
        ]

counter = 1
temp_section = []
short_section = []
last_section_short = False


sum_lines = 0
avgs_letters = 0

for i in range(len(lines)):
    for d in deletions: 
        lines[i] = re.sub(d," ",lines[i])

    line = re.sub(r'[^a-zA-Z\sěščřžýáíéúůóťďňĚŠČŘŽÝÁÍÉÚŮÓŤĎŇ,.()]+', ' ', lines[i])
    line = re.sub(r'\s+', ' ', line)

    if not re.sub(r"[.,()]", '', line).strip() and len(temp_section) > 0:    # if the line contains just white spaces
        # print("lines in a section: ", len(temp_section), "index: ", counter)

        lang, prob = detect_language_with_langdetect(" ".join(temp_section))
        if lang != "cs" and lang != "sk" and lang != "sl":
            continue

        if len(temp_section) > MAX_LINES:
            if last_section_short:
                (sum_lines, counter, short_section, last_section_short) = section_join(sum_lines, short_section, temp_section, counter, MAX_LINES, MIN_LINES)
            else:
                (sum_lines, counter) = recursive_section_split(sum_lines, temp_section, counter, MAX_LINES, MIN_LINES)
            # print(counter)

        elif len(temp_section) < MIN_LINES:
            if last_section_short:
                (sum_lines, counter, short_section, last_section_short) = section_join(sum_lines, short_section, temp_section, counter, MAX_LINES, MIN_LINES)
            else:
                last_section_short = True
                short_section = temp_section.copy()

        else:
            if last_section_short:
                (sum_lines, counter, short_section, last_section_short) = section_join(sum_lines, short_section, temp_section, counter, MAX_LINES, MIN_LINES)
            else:
                with open("DATA\\Velky_zpevnik\\VZ_sections\\zpevnik_" + str(counter) + ".txt", "w", encoding="utf-8") as new_section:
                    new_section.writelines(temp_section)
                counter += 1
                sum_lines += len(temp_section)

        temp_section.clear()
        print(counter - 1)
        print("avg lines in section:", sum_lines / (counter - 1))
        print("letters/line:", avgs_letters / sum_lines)
        

    if re.sub(r"[.,()]", '', line).strip():
        while len(line) > 0 and re.match(r'[\s,.()]+', line[-1]):
            line = line[:-1]
        while re.match(r'[\s,.()]+', line[0]):
            line = line[1:]

        avgs_letters += len(line)

        if len(line) > TRY_SPLIT_LINE_LEN:
            split_line : list[str] = re.split(r"[.,()]", line)
            cleaned_split_line = []
            for split_i in range(len(split_line)):
                if not split_line[split_i].strip():       # there are empty line parts
                    continue

                while re.match(r'[\s,.()]+', split_line[split_i][0]):
                    split_line[split_i] = split_line[split_i][1:]
                while re.match(r'[\s,.()]+', split_line[split_i][-1]):
                    split_line[split_i] = split_line[split_i][:-1]

                cleaned_split_line.append(split_line[split_i])
            
            for split_i in range(len(cleaned_split_line)):
                line_part = cleaned_split_line[split_i]

                if len(line_part) > MAX_LINE_LEN:
                    pass
                elif len(line_part) < MIN_LINE_LEN:
                    if split_i + 1 < len(cleaned_split_line):   # there is a following split section
                        if len(line_part) + len(cleaned_split_line[split_i + 1]) + 2 < MAX_LINE_LEN:    # including ", " it is short enough
                            cleaned_split_line[split_i + 1] = line_part + ", " + cleaned_split_line[split_i + 1]
                else:
                    temp_section.append(line_part + "\n")

        elif len(line) < 5:
            pass

        else:
            while re.match(r'[\s,.()]+', line[0]):
                line = line[1:]
            while re.match(r'[\s,.()]+', line[-1]):
                line = line[:-1]

            temp_section.append(re.sub(r"[()]", '', line) + "\n")




