import os
import Imports.tagger as tagger
import math
from phoneme_repetition_similarity import phoneme_distinct2
import matplotlib.pyplot as plt
import json


directory = 'DATA/Aligned_HT'
 
HT_dataset = {}

rt_cs = tagger.RhymeTagger()
rt_cs.load_model("cs", verbose=False)



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
        print(temp_section_1)

    # Recursively split or write into a file
    if len(temp_section_2) > max_lines:
        counter = recursive_section_split(temp_section_2, counter, max_lines)
    else:
        counter = write_section(temp_section_2, counter)
        print(temp_section_2)

    return counter


def write_section(section: list[str], section_i: int):
    HT_dataset[language][moviename][songname][section_i] = {"lyrics": section.copy()}
    return section_i + 1















section_lengths = [0] * 50





for filename in os.listdir(directory):
    moviename = filename[:-10]
    songname = filename[-9:-7]
    language = filename[-6:-4]

    if language not in HT_dataset:
        HT_dataset[language] = {}

    if moviename not in HT_dataset[language]:
        HT_dataset[language][moviename] = {}

    if songname not in HT_dataset[language][moviename]:
        HT_dataset[language][moviename][songname] = {}

    f = os.path.join(directory, filename)
    if os.path.isfile(f):
        print("\n", f)

    with open(f, "r", encoding="utf-8") as en_file:
        song = en_file.readlines()

    

    section = []
    section_i = 0

    for i in range(len(song)):
        line = song[i]
        stripped_line = line.strip()
        if not stripped_line:
            section_lengths[len(section)] += 1

            # dialogue
            if len(section) <= 1:
                print("delete:        ", section)
            
            elif len(section) > 10:
                if language == "cs":
                    print(section_i)
                    section_i = recursive_section_split(section, section_i, 10, 3)
                    
                else:
                    while len(section) > 0:
                        wanted_len = len(HT_dataset["cs"][moviename][songname][section_i]["lyrics"])
                        section_i = write_section(section[:wanted_len], section_i)
                        section = section[wanted_len:]

            else:
                section_i = write_section(section, section_i)
                
            section.clear()
            continue
        
        section.append(line)
        

plt.plot(section_lengths)
plt.show()


with open("DATA\\Velky_zpevnik\\HT.json", "w", encoding='utf-8') as json_file:
    json.dump(HT_dataset, json_file, ensure_ascii=False)
        
    


    
