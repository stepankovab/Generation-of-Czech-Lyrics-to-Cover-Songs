from syllabator import syllabify
from nltk.tokenize import word_tokenize


def divide_line_to_words_and_syllables(line : str):
    line = line.lower()
    words = [word for word in word_tokenize(line, "czech") if str.isalpha(word)]

    syllabified_words = []

    last_prep = ""
    for word in words:
        if word in "vkzs":
            last_prep = word
            continue

        if last_prep != "":
            word = last_prep + " " + word
            last_prep = ""

        syllabified_words.append(syllabify(word, "cs"))

    print(6)




divide_line_to_words_and_syllables("Hej ted to zkusim, jestli to je v poradku.")