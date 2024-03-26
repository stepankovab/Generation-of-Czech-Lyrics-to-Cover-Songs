from eval.tagger import RhymeTagger
from eval.rhyme_finder import RhymeFinder
from eval.same_word_tagger import SameWordRhymeTagger
from HT_loader import HT_loader
from evaluator import Evaluator
import matplotlib.pyplot as plt
import numpy as np

ev = Evaluator()


def _fill_in_none_rhymes(rhymes):
    """
    Rewrites numeric rhyme scheme into capital letters. Fills in different letters for each None tag.

    Parameters:
    ----------
    rhymes: list of int or None describing the rhyme scheme

    Returns:
    ---------
    rhyme scheme in capital letters
    """
    max_rhyme_ref = 0
    none_ids = []
    for rhyme_i in range(len(rhymes)):
        if isinstance(rhymes[rhyme_i], int):
            if rhymes[rhyme_i] > max_rhyme_ref:
                max_rhyme_ref = rhymes[rhyme_i]
            # convert to capital letters, start with A
            rhymes[rhyme_i] = chr(64 + rhymes[rhyme_i])
        else:
            none_ids.append(rhyme_i)

    for none_i in none_ids:
        max_rhyme_ref += 1
        rhymes[none_i] = chr(64 + max_rhyme_ref)

    return rhymes


def print_histogram(data1, data2, name1, name2, file_name, lang):
    bins = np.linspace(0, 1, 10)
    plt.hist(data1, bins, density=True, alpha=0.5, label=name1, edgecolor='black')
    plt.hist(data2, bins, density=True, alpha=0.5, label=name2, edgecolor='black')
    plt.xlabel('Results')
    plt.ylabel('Frequency')
    plt.title(f"Comparison of {name1} and {name2} in {lang}")
    plt.legend(loc='upper center')
    plt.savefig(f"{file_name}_{lang}.png")
    plt.show()



LANG = "en"

rt = RhymeTagger()
rt.load_model(LANG, verbose=False)
rf = RhymeFinder(LANG)
ht = HT_loader("DATA", LANG)
swrt = SameWordRhymeTagger(LANG)

tagfin = []
fintag = []
tagwor = []
wortag = []
finwor = []
worfin = []

for text in ht:
    text = text.split(",")

    print(text[0])

    # print()
    # for line in text:
    #     print(line)
    # print()

    a = _fill_in_none_rhymes(rt.tag(poem=text, output_format=3))
    b = rf.tag(text)
    c = swrt.tag(text, output_format=3)

    # print(a)
    # print(b)
    # print(c)

    ab = ev.get_rhyme_scheme_agreement(a, b)
    ba = ev.get_rhyme_scheme_agreement(b, a)
    ac = ev.get_rhyme_scheme_agreement(a, c)
    ca = ev.get_rhyme_scheme_agreement(c, a)
    bc = ev.get_rhyme_scheme_agreement(b, c)
    cb = ev.get_rhyme_scheme_agreement(c, b)

    # print("tagger -> finder", ab)
    # print("finder -> tagger", ba)
    # print("tagger -> wordsT", ac)
    # print("wordsT -> tagger", ca)
    # print("finder -> wordsT", bc)
    # print("wordsT -> finder", cb)

    tagfin.append(ab)
    fintag.append(ba)
    tagwor.append(ac)
    wortag.append(ca)
    finwor.append(bc)
    worfin.append(cb)

print_histogram(tagfin, fintag, 'rhymefinder', 'rhymetagger', 'fintag', LANG)
print_histogram(tagwor, wortag, 'same-word rhymetagger', 'rhymetagger', 'tagwor', LANG)
print_histogram(finwor, worfin, 'same-word rhymetagger', 'rhymefinder', 'finwor', LANG)

print()
print("tagger -> finder avg:", sum(tagfin)/len(tagfin))
print("finder -> tagger avg:", sum(fintag)/len(fintag))
print("tagger -> wordsT avg:", sum(tagwor)/len(tagwor))
print("wordsT -> tagger avg:", sum(wortag)/len(wortag))
print("finder -> wordsT avg:", sum(finwor)/len(finwor))
print("wordsT -> finder avg:", sum(worfin)/len(worfin))




# EN compare  
# neznamy slova ignoruje no...


# tagger -> finder avg: 0.9050766195456461
# tagger -> wordsT avg: 0.9964601769911504
# wordsT -> tagger avg: 0.7821958652140851
# finder -> wordsT avg: 0.9030145407689655
# wordsT -> finder avg: 0.8916240695073333

# CS compare:
# finder je lepsi, ale myslim ze kdyby se do taggeru pridali shodny slova tak mozna bude lepsi

# tagger -> finder avg: 0.8664608793369856
# finder -> tagger avg: 0.8138087576140675
# tagger -> wordsT avg: 0.9988200589970501
# wordsT -> tagger avg: 0.8551193349423439
# finder -> wordsT avg: 0.9555794353139485
# wordsT -> finder avg: 0.8692948447815706



