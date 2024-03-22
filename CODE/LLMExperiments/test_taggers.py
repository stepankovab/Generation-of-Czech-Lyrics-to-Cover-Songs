
from eval.tagger import RhymeTagger
from eval.rhyme_finder import RhymeFinder
from HT_loader import HT_loader
from evaluator import Evaluator

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

cs_rt = RhymeTagger()
cs_rt.load_model("cs", verbose=False)

cs_rf = RhymeFinder("cs")

cs_ht = HT_loader("DATA", "cs")

for text in cs_ht:
    text = text.split(",")

    print()
    for line in text:
        print(line)
    print()

    a = _fill_in_none_rhymes(cs_rt.tag(poem=text, output_format=3))
    b = cs_rf.tag(text)

    print(a)
    print(b)

    print("original tagger", ev.get_rhyme_scheme_agreement(a, b))
    print("original finder", ev.get_rhyme_scheme_agreement(b, a))


