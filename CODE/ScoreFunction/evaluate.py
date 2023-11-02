from syllable_count_distance import get_line_syllable_count_distance
from phoneme_repetition_similarity import get_phoneme_repetition_similarity
from musical_structure_distance import get_musical_structure_distance
from semantic_similarity import get_semantic_similarity
from rhyme_scheme_distance import get_rhyme_scheme_mapping_distance
from phoneme_edit_distance import get_phoneme_edit_distance
from lyrics_aligner import get_aligned_translation
from extract_syl_mapping import get_phoneme_syllable_mapping_distance
import os
import lzma
import pickle


def score_function(czech_sections : list[list[str]], english_sections : list[list[str]]) -> float:
    """
    Expects line aligned lyrics

    Sections divided by empty lines
    """

    score = []

    print("The Metric modified for czech and english texts:")

    line_syllable_count_distance = get_line_syllable_count_distance(czech_sections, english_sections)
    print("line syllable count distance: ", line_syllable_count_distance)
    score.append(line_syllable_count_distance)

    phoneme_repetition_correlation = get_phoneme_repetition_similarity(czech_sections, english_sections)
    print("Phoneme repetition correlation: ", phoneme_repetition_correlation)
    score.append(phoneme_repetition_correlation)

    musical_structure_distance = get_musical_structure_distance(czech_sections, english_sections)
    print("musical structure distance: ", musical_structure_distance)
    score.append(musical_structure_distance)

    semantic_similarity = get_semantic_similarity(czech_sections, english_sections)
    print("semantic similarity: ", semantic_similarity)
    score.append(semantic_similarity)


    
    print("\nMy own experiments:")

    rhyme_scheme_mapping_distance = get_rhyme_scheme_mapping_distance(czech_sections, english_sections)
    print("rhyme scheme mapping distance: ", rhyme_scheme_mapping_distance)
    score.append(rhyme_scheme_mapping_distance)

    phoneme_edit_distance = get_phoneme_edit_distance(czech_sections, english_sections)
    print("phoneme edit distance: ", phoneme_edit_distance)
    score.append(phoneme_edit_distance)

    phoneme_syllable_mapping_distance = get_phoneme_syllable_mapping_distance(czech_sections, english_sections)
    print("phoneme syllable mapping distance: ", phoneme_syllable_mapping_distance)
    score.append(phoneme_syllable_mapping_distance)


    return score



scores = {}
scores["HT"] = []
scores["cz_to_en"] = []
scores["en_to_cz"] = []

filenames = os.listdir("C:/Users/barca/MOJE/UMEA/NLP-Umea/FinalProject/DATA/Aligned_HT")

czech_og = []
counter = 0

for filename in filenames:

    og_sections = []
    with open("C:/Users/barca/MOJE/UMEA/NLP-Umea/FinalProject/DATA/Aligned_HT/" + filename , "r", encoding="utf-8") as f:
        lines = f.readlines()
        temp = []
        for line in lines:
            line = line[:-1]
            if line == "":
                og_sections.append(temp.copy())
                temp.clear()
            else:
                temp.append(line)   

    if filename[-6:] == "en.txt":
        print("\n\n" + filename[:-7] + " from english to czech:")
        czech_sections = get_aligned_translation(og_sections, "cz")
        scores["en_to_cz"].append(score_function(czech_sections, og_sections))
        
        print("\n\n" + filename[:-7] + " - the original HT lyrics:")
        scores["HT"].append(score_function(czech_og, og_sections))
    else:
        print("\n\n" + filename[:-7] + " from czech to english:")
        english_sections = get_aligned_translation(og_sections, "en")
        scores["cz_to_en"].append(score_function(og_sections, english_sections))
        czech_og = og_sections

    counter += 1



# Serialize the model.
with lzma.open("C:/Users/barca/MOJE/UMEA/NLP-Umea/FinalProject/DATA/results.dict", "wb") as model_file:
    pickle.dump(scores, model_file)


        