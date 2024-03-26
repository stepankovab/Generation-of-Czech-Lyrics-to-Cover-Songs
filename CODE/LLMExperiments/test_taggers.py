from eval.tagger import RhymeTagger
from eval.rhyme_finder import RhymeFinder
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

LANG = "en"

rt = RhymeTagger()
rt.load_model(LANG, verbose=False)
rf = RhymeFinder(LANG)
ht = HT_loader("DATA", LANG)

tagfin = []
fintag = []

for text in ht:
    text = text.split(",")

    print(text[0])

    # print()
    # for line in text:
    #     print(line)
    # print()

    a = _fill_in_none_rhymes(rt.tag(poem=text, output_format=3))
    b = rf.tag(text)

    # print(a)
    # print(b)

    c = ev.get_rhyme_scheme_agreement(a, b)
    d = ev.get_rhyme_scheme_agreement(b, a)

    # print("tagger -> finder", c)
    # print("finder -> tagger", d)

    tagfin.append(c)
    fintag.append(d)


bins = np.linspace(0, 1, 10)

plt.hist(tagfin, bins, density=True, alpha=0.5, label='rhymefinder', edgecolor='black')
plt.hist(fintag, bins, density=True, alpha=0.5, label='rhymetagger', edgecolor='black')
plt.xlabel('Results')
plt.ylabel('Frequency')
plt.title(f"Comparison of tagger and finder in {LANG}")
plt.legend(loc='upper center')
plt.savefig(f"tagfin_{LANG}.png")
plt.show()

print()
print("tagger -> finder avg:", sum(tagfin)/len(tagfin))
print("finder -> tagger avg:", sum(fintag)/len(fintag))




# EN compare  
# neznamy slova ignoruje no...

# CS compare:
# finder je lepsi, ale myslim ze kdyby se do taggeru pridali shodny slova tak mozna bude lepsi



