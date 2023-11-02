from syllabator import syllabify

def get_line_syllable_count_distance(czech_sections : list[list[str]], english_sections : list[list[str]]) -> float:
    '''
    Get the distance of syllable counts averaged over lines.
    '''
    
    czech_lines = []
    for section in czech_sections:
        czech_lines.extend(section)
    
    english_lines = []
    for section in english_sections:
        english_lines.extend(section)
    
    if len(czech_lines) != len(english_lines):
        raise ValueError("The lyrics differ in number of lines.")
    
    cz_syl = [len(syllabify(line, "cz")) for line in czech_lines]
    en_syl = [len(syllabify(line, "en")) for line in english_lines]


    n = len(czech_lines)
    distance = 0

    for i in range(n):
        distance += (abs(cz_syl[i] - en_syl[i]) / max(cz_syl[i], 1)) + (abs(cz_syl[i] - en_syl[i]) / max(en_syl[i], 1))

    distance /= (2 * n)

    return distance