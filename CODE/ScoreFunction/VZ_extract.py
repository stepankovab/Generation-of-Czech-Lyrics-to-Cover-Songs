import re
import math
import json
from langdetect import detect_langs
from VZ_globals import set_of_sections, dataset_dict
from phoneme_repetition_similarity import phoneme_distinct2

VERBOSE = False

MAX_LINES = 8
MIN_LINES = 3

MAX_LINE_LEN = 55
TRY_SPLIT_LINE_LEN = 25
MIN_LINE_LEN = 5

def check_for_prob_of_czech(line : str) -> bool: 
    """
    Detects if there is a probability of the text being Czech.

    Parameters
    -----------
    line: string of text to be checked

    Returns
    ----------
    boolean if there is a probability of the text being Czech
    """
    try: 
        langs = detect_langs(line) 
        cs_true = False
        for item in langs: 
            if item.lang == "cs":
                cs_true = True
        return cs_true
    except: return False

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
    # Split longer sections in half without evaluation
    if len(temp_section) > 50:
        log("section over 50 lines: " + str(len(temp_section)))
        temp_split_i = len(temp_section)//2
        return (temp_section[:temp_split_i], temp_section[temp_split_i:])

    # Split shorter sections based on phoneme distinct2 score
    best_split_value = math.inf
    best_split_i = 0

    for split_i in range(min_lines, len(temp_section) - min_lines + 1):
        phon2_1 = phoneme_distinct2(temp_section[:split_i], "cz")
        phon2_2 = phoneme_distinct2(temp_section[split_i:], "cz")

        if phon2_1 + phon2_2 < best_split_value: # and -> so its rozumny sekce
            best_split_value = phon2_1 + phon2_2
            best_split_i = split_i

    return (temp_section[:best_split_i], temp_section[best_split_i:])

def recursive_section_split(temp_section : list[str], counter : int, max_lines : int = 10, min_lines : int = 2) -> int:
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

    # Recursively split or write into a file
    if len(temp_section_1) > max_lines:
        counter = recursive_section_split(temp_section_1, counter, max_lines)
    else:
        counter = write_section(temp_section_1, counter)

    # Recursively split or write into a file
    if len(temp_section_2) > max_lines:
        counter = recursive_section_split(temp_section_2, counter, max_lines)
    else:
        counter = write_section(temp_section_2, counter)

    return counter


def section_join(short_section : list[str], next_section : list[str], counter : int, max_lines : int = 10, min_lines : int = 2) -> tuple[int, list[str], bool] :
    """
    Appends the short section to the previous or following section.
    Deals with the aftermath of overflowing or underflowing.

    Parameters
    -----------
    short_section: The last short section
    next_section: Section following the short section
    counter: Number of the section saved into file
    max_lines: Upper bound on the line count per section
    min_lines: Lower bound on the line count per section
    
    Returns
    ---------
    counter: int - after saving sections
    return_short: list[str] - new short section or empty list
    return_short_bool: bool - True if new short section exists
    """    
    return_short = []
    return_short_bool = False

    # Load the section before the short section
    prev_section = dataset_dict[counter - 1]["lyrics"]

    # Pair up sections based on phoneme distinct2
    phon2_prev = phoneme_distinct2(prev_section + short_section, "cz")
    phon2_next = phoneme_distinct2(short_section + next_section, "cz")

    if phon2_prev < phon2_next:
        first_section = prev_section + short_section
        second_section = next_section
    else:
        first_section = prev_section
        second_section = short_section + next_section

    # Recursively split or write into a file
    if len(first_section) > max_lines:
        counter = recursive_section_split(first_section, counter - 1, max_lines, min_lines)
    else:
        counter = write_section(first_section, counter - 1, rewrites_previous = True, prev_section = prev_section)
    
    # Recursively split, save as a new short section, or write into a file
    if len(second_section) > max_lines:
        counter = recursive_section_split(second_section, counter, max_lines, min_lines)
    elif len(second_section) < min_lines:
        return_short = second_section
        return_short_bool = True
    else:
        counter = write_section(second_section, counter)

    return (counter, return_short, return_short_bool)

def write_section(section : list[str], counter : int, rewrites_previous : bool = False, prev_section : list[str]|None = None) -> int:
    """
    Creates a file called "zpevnik_*counter*" and saves the section.

    Parameters:
    --------
    section: lines to be saved as a section
    counter: order number of the section
    rewrites_previous: rewrites previously written section
    prev_section: !!fill in only if rewriting previous section!!

    Returns:
    --------
    increased counter
    """
    joined_section = " ".join(section)
    cs_true = check_for_prob_of_czech(joined_section)
    if not cs_true:
        return counter

    if joined_section in set_of_sections:
        if rewrites_previous:
            return counter + 1
        else:
            return counter
    else:
        set_of_sections.add(joined_section)
        if rewrites_previous:
            set_of_sections.remove(" ".join(prev_section))

    log(counter)
    dataset_dict[counter] = {"lyrics": section.copy()}
    return counter + 1


def log(message : str):
    """
    Logs a message into the console

    Parameters
    ----------
    message: to be logged
    """
    if VERBOSE:
        print(message)




with open("DATA\\Velky_zpevnik\\velkyzpevnik.txt", "r", encoding="utf-8") as file:
    lines = file.readlines()


deletions = [
    r"[0-9]+[x]*",
    r"\br\b",
    r"\brf\b",
    r"\bref\b",
    r"\brefr\b",
    r"\brefrén\b",
    r"\brefrain\b",
    r"\b[cdefgahb](is)*mi\b",
    r"\b[cdfg]is\b",
    r"\b[ae]s\b",
    r"\b[hb]es\b",
    r"\b[cdefghb][éáb#]*\b",
    r"\bx\b"    
        ]

line_deletion_keys = [
    r"akord",
    r"\bCD",
    r"predehra",
    r"předehra",
    r"mezihra",
    r"kápo",
    r"\.\.\.",
    r"hudba",
    r"text",
    r"intro",
    r"capo",
    r"II",
    r"sólo",
    r"sbor",
    r"nápěv",
    r"sloka"
        ]

counter = 1
temp_section = []
short_section = []
last_section_short = False

set_of_sections = set()

for i in range(len(lines)):
    line = lines[i].lower()

    # Delete lines containing line deletion keys
    skip_line = False
    for ldk in line_deletion_keys:
        if re.match(ldk, line):
            skip_line = True
            break
    if skip_line:
        print(counter)
        print(line)
        continue
        

    # Delete unwanted words and symbols from the line
    for d in deletions: 
        line = re.sub(d,"",line)
    line = re.sub(r'[^a-zA-Z\sěščřžýáíéúůóťďňĚŠČŘŽÝÁÍÉÚŮÓŤĎŇ,\.()]+', '', line)
    line = re.sub(r'\s+', ' ', line)
    
    # Found a break in lines, try to write the section
    if not re.sub(r"[\.,()]", '', line).strip() and len(temp_section) > 0:

        # Section is too long
        if len(temp_section) > MAX_LINES:
            if last_section_short:
                (counter, short_section, last_section_short) = section_join(short_section, temp_section, counter, MAX_LINES, MIN_LINES)
            else:
                counter = recursive_section_split(temp_section, counter, MAX_LINES, MIN_LINES)

        # Section is too short
        elif len(temp_section) < MIN_LINES:
            if last_section_short:
                (counter, short_section, last_section_short) = section_join(short_section, temp_section, counter, MAX_LINES, MIN_LINES)
            else:
                last_section_short = True
                short_section = temp_section.copy()

        # Section has a correct length
        else:
            if last_section_short:
                (counter, short_section, last_section_short) = section_join(short_section, temp_section, counter, MAX_LINES, MIN_LINES)
            else:
                counter = write_section(temp_section, counter)

        temp_section.clear()
        
    # Adding a line to temp section:
        
    if not re.sub(r'[\s,\.()]+', "", line).strip():
        continue
    while len(line) > 0 and re.match(r'[\s,\.()]+', line[-1]):
        line = line[:-1]
    while re.match(r'[\s,\.()]+', line[0]):
        line = line[1:]

    # if longer than SPLIT_LINE_LEN, try splitting the line
    if len(line) > TRY_SPLIT_LINE_LEN:
        split_line : list[str] = re.split(r"[\.,()]", line)
        cleaned_split_line = []

        # Clean up the split parts
        for split_i in range(len(split_line)):
            if not split_line[split_i].strip():
                continue
            while re.match(r'[\s,\.()]+', split_line[split_i][0]):
                split_line[split_i] = split_line[split_i][1:]
            while re.match(r'[\s,\.()]+', split_line[split_i][-1]):
                split_line[split_i] = split_line[split_i][:-1]

            cleaned_split_line.append(split_line[split_i])
        
        # Combine and add split parts to the temp section
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

    # ignore too short lines
    elif len(line) < MIN_LINE_LEN:
        pass

    # simply add lines of good length
    else:
        while re.match(r'[\s,\.()]+', line[0]):
            line = line[1:]
        while re.match(r'[\s,\.()]+', line[-1]):
            line = line[:-1]

        temp_section.append(re.sub(r"[()]", '', line) + "\n")



print(counter)                  # 77484
print(len(set_of_sections))     # 78909

with open("DATA\\Velky_zpevnik\\VZ_pure.json", "w", encoding='utf-8') as json_file:
    json.dump(dataset_dict, json_file, ensure_ascii=False)
