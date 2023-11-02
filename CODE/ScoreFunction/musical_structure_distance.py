from phoneme_repetition_similarity import phoneme_distinct2
import math

def musical_dissimilarity(section1 : list[str], section2 : list[str], language : str) -> float:
    """
    language: the language of the section ['cz', 'en']
    """
    return phoneme_distinct2(section1 + section2, language) + abs(phoneme_distinct2(section1, language) - phoneme_distinct2(section2, language))


def get_musical_structure_distance(czech_sections : list[list[str]], english_sections : list[list[str]]) -> float:
    '''
    Get the distance between the musical structure of the lyrics.
    '''
    
    if len(czech_sections) != len(english_sections):
        raise ValueError("The lyrics differ in number of sections.")
    
    n_sections = len(czech_sections)

    square_diss_sum = 0

    for i in range(n_sections):
        for j in range(n_sections):
            square_diss_sum += (musical_dissimilarity(czech_sections[i], czech_sections[j], "cz") - musical_dissimilarity(english_sections[i], english_sections[j], "en"))**2

    return 1/n_sections**2 * math.sqrt(square_diss_sum)