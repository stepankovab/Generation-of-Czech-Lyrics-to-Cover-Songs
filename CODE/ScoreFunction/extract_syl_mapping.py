from syllabator import syllabify, _create_word_mask
import os
import lzma
import pickle


def get_syllable_parts(mask, syl):
    syl_b = ""
    syl_m = ""
    syl_e = ""

    last_zero = False
    end = False
    for i in range(len(mask)):
        if end and mask[i] == "K":
            if last_zero:
                syl_e += syl[i-1]
                last_zero = False
            syl_e += syl[i]
        
        elif mask[i] == "K":
            if last_zero:
                syl_b += syl[i-1]
                last_zero = False
            syl_b += syl[i]

        elif mask[i] == "0":
            last_zero = True

        else:
            if last_zero:
                syl_m += syl[i]
                last_zero = False
            syl_m += syl[i]
            end = True

    if last_zero:
        syl_e += syl[-1]

    return syl_b, syl_m, syl_e



def get_syllable_part_sound_mapping_dict():
    pair_dict = {}

    filenames = os.listdir("C:/Users/barca/MOJE/UMEA/NLP-Umea/FinalProject/DATA/Aligned_HT")[:50]

    for filename in filenames:
        if filename[-6:] == "en.txt":
            continue

        filename_root = filename[:-6]

        with open("C:/Users/barca/MOJE/UMEA/NLP-Umea/FinalProject/DATA/Aligned_HT/" + filename_root + "cs.txt" , "r", encoding="utf-8") as f:
            czech_lines = f.readlines()
            czech_lines = [line[:-1] for line in czech_lines if line != "\n"]
            
        with open("C:/Users/barca/MOJE/UMEA/NLP-Umea/FinalProject/DATA/Aligned_HT/" + filename_root + "en.txt" , "r", encoding="utf-8") as f:
            english_lines = f.readlines()
            english_lines = [line[:-1] for line in english_lines if line != "\n"]

        if len(czech_lines) != len(english_lines):
            raise ValueError("The lyrics differ in number of lines.")

        n_lines = len(czech_lines)

        for n_l in range(n_lines):
            en_syllables = syllabify(english_lines[n_l], "en")
            cs_syllables = syllabify(czech_lines[n_l], "cs")

            if len(en_syllables) != len(cs_syllables):
                continue

            n_syllables = len(en_syllables)

            for n_s in range(n_syllables):
                en_syl = en_syllables[n_s]
                cs_syl = cs_syllables[n_s]

                en_mask = _create_word_mask(en_syl)
                cs_mask = _create_word_mask(cs_syl)

                en_syl_b, en_syl_m, en_syl_e = get_syllable_parts(en_mask, en_syl)    
                cs_syl_b, cs_syl_m, cs_syl_e = get_syllable_parts(cs_mask, cs_syl)

                if (en_syl_b, cs_syl_b) not in pair_dict:
                    pair_dict[(en_syl_b, cs_syl_b)] = 0

                if (en_syl_m, cs_syl_m) not in pair_dict:
                    pair_dict[(en_syl_m, cs_syl_m)] = 0

                if (en_syl_e, cs_syl_e) not in pair_dict:
                    pair_dict[(en_syl_e, cs_syl_e)] = 0

                pair_dict[(en_syl_b, cs_syl_b)] += 1
                pair_dict[(en_syl_m, cs_syl_m)] += 1
                pair_dict[(en_syl_e, cs_syl_e)] += 1

    # pair_dict = sorted(pair_dict.items(), key=lambda x: x[1], reverse=False)

    # Serialize the model.
    with lzma.open("C:/Users/barca/MOJE/UMEA/NLP-Umea/FinalProject/DATA/pair.dict", "wb") as model_file:
        pickle.dump(pair_dict, model_file)


def get_phoneme_syllable_mapping_distance(czech_sections, english_sections):
    distance = 0

    with lzma.open("C:/Users/barca/MOJE/UMEA/NLP-Umea/FinalProject/DATA/pair.dict", "rb") as model_file:
        pair_dict = pickle.load(model_file)

    czech_lines = []
    for section in czech_sections:
        czech_lines.extend(section)
    
    english_lines = []
    for section in english_sections:
        english_lines.extend(section)

    if len(czech_lines) != len(english_lines):
            raise ValueError("The lyrics differ in number of lines.")

    n_lines = len(czech_lines)

    for n_l in range(n_lines):
        en_syllables = syllabify(english_lines[n_l], "en")
        cs_syllables = syllabify(czech_lines[n_l], "cs")

        if len(en_syllables) < len(cs_syllables):
            while len(en_syllables) != len(cs_syllables):
                en_syllables.append("")
        
        if len(en_syllables) > len(cs_syllables):
            while len(en_syllables) != len(cs_syllables):
                cs_syllables.append("")

        n_syllables = len(en_syllables)

        for n_s in range(n_syllables):
            en_syl = en_syllables[n_s]
            cs_syl = cs_syllables[n_s]

            en_mask = _create_word_mask(en_syl)
            cs_mask = _create_word_mask(cs_syl)

            en_syl_b, en_syl_m, en_syl_e = get_syllable_parts(en_mask, en_syl)    
            cs_syl_b, cs_syl_m, cs_syl_e = get_syllable_parts(cs_mask, cs_syl)

            b_count = 0
            m_count = 0
            e_count = 0

            if (en_syl_b, cs_syl_b) in pair_dict:
                b_count = pair_dict[(en_syl_b, cs_syl_b)]

            if (en_syl_m, cs_syl_m) in pair_dict:
                m_count = pair_dict[(en_syl_m, cs_syl_m)]

            if (en_syl_e, cs_syl_e) in pair_dict:
                e_count = pair_dict[(en_syl_e, cs_syl_e)]

            distance += 1 / max(b_count , 5)
            distance += 1 / max(m_count , 5)
            distance += 1 / max(e_count , 5)
        
        distance /= n_syllables

    return distance

    



# get_syllable_part_sound_mapping_dict()

