from Imports.phon_czech import ipa_czech
import eng_to_ipa as ipa
import re


strong_similarity = {
    ('aː', 'a'),
    ('ɛː', 'ɛ'),
    ('ɛː', 'ə'),
    ('ɛː', 'æ'),
    ('iː', 'i'),
    ('iː', 'ɪ'),
    ('oː', 'ɔ'),
    ('uː', 'u'),
    ('uː', 'ʊ'),
    ('u', 'ʊ'),
    ('yː', 'y'),
    ('ə', 'a'),
    ('ə', 'aː'),
    ('ə', 'ɛ'),
    ('ɛ', 'æ'),
    ('m', 'n'),
    ('ŋ', 'n'),
    ('ŋ', 'm'),
    ('t͡s', 'c'),
    ('ð', 'd'),
    ('o', 'ɔ')
}

vocals = {'aː', 'a', 'ɛː', 'ɛ', 'ə', 'æ', 'iː', 'i', 'ɪ', 'oː', 'ɔ', 'o', 'uː', 'u', 'ʊ', 'yː', 'y'}


def get_phoneme_edit_distance(czech_sections : list[list[str]], english_sections : list[list[str]]) -> float:
    
    czech_lines = []
    for section in czech_sections:
        czech_lines.extend(section)
    
    english_lines = []
    for section in english_sections:
        english_lines.extend(section)

    if len(czech_lines) != len(english_lines):
        raise ValueError("The lyrics differ in number of lines.")

    n_lines = len(czech_lines)

    distance_sum = 0

    for i in range(n_lines):
        czech_pho = ipa_czech(czech_lines[i]).split() + ["^"]
        english_pho = [*re.sub('[ˈ\s]', '', ipa.convert(english_lines[i], stress_marks='primary')) + "^"]

        dist = edit_distance(czech_pho, english_pho, len(czech_pho), len(english_pho))

        distance_sum += (dist/( 2 * len(czech_pho)) + dist/(2 * len(english_pho)))

    return(distance_sum/n_lines)



def edit_distance(str1, str2, m, n):
    # Create a table to store results of subproblems
    dp = [[0 for x in range(n + 1)] for x in range(m + 1)]

    # Fill d[][] in bottom up manner
    for i in range(m + 1):
        for j in range(n + 1):
 
            # If first string is empty, only option is to
            # insert all characters of second string
            if i == 0:
                dp[i][j] = j    # Min. operations = j
 
            # If second string is empty, only option is to
            # remove all characters of second string
            elif j == 0:
                dp[i][j] = i    # Min. operations = i
 
            # If last characters are same, ignore last char
            # and recur for remaining string
            elif str1[i-1] == str2[j-1]:
                dp[i][j] = dp[i-1][j-1]

            elif (str1[i-1], str2[j-1]) in strong_similarity or (str2[j-1], str1[i-1]) in strong_similarity:
                dp[i][j] = dp[i-1][j-1] + 0.1


            elif str1[i-1] in vocals and str2[j-1] in vocals:
                dp[i][j] = dp[i-1][j-1] + 0.4


            elif str1[i-1] not in vocals and str2[j-1] not in vocals:
                dp[i][j] = dp[i-1][j-1] + 0.7        

 
            # If last character are different, consider all
            # possibilities and find minimum
            else:
                dp[i][j] = 1 + min(dp[i][j-1],        # Insert
                                   dp[i-1][j],        # Remove
                                   dp[i-1][j-1])    # Replace
 
    return dp[m][n]