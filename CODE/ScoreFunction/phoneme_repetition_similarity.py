from Imports.phon_czech import ipa_czech
import eng_to_ipa as ipa
import re
import scipy.stats as stats


def phoneme_distinct2(section : list[str], language : str) -> float:
    """
    language: the language of the section ['cz', 'en']
    """
    bigram_dict = {}
    bigram_count = 0

    for i in range(len(section)):
        if language == "cz":
            phonemes = ipa_czech(section[i]).split() + ["^"]
        if language == "en":
            phonemes = re.sub('[ˈ\s]', '', ipa.convert(section[i], stress_marks='primary')) + "^"

        for p in range(len(phonemes) - 1):
            bigram = phonemes[p] + phonemes[p + 1]
            bigram_count += 1

            if bigram not in bigram_dict:
                bigram_dict[bigram] = 0
            
            bigram_dict[bigram] += 1

    return len(bigram_dict)/ max(bigram_count,1)


def get_phoneme_repetition_similarity(czech_sections : list[list[str]], english_sections : list[list[str]]) -> float:
    
    if len(czech_sections) != len(english_sections):
        raise ValueError("The lyrics differ in number of sections.")
    
    n_sections = len(czech_sections)

    czech_distinct2 = []
    english_distinct2 = []

    for s in range(n_sections):
        czech_distinct2.append(phoneme_distinct2(czech_sections[s], "cz"))
        english_distinct2.append(phoneme_distinct2(english_sections[s], "en"))

    rho, p = stats.spearmanr(czech_distinct2, english_distinct2)

    return rho