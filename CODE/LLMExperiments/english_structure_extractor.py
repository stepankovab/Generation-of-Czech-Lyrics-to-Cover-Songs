from eval.en_syllabator import syllabify
# from rhymetagger import RhymeTagger
import requests
from keybert import KeyBERT


class SectionStructure:
    syllables = []
    rhyme_scheme = []
    keywords = []
    num_lines: int

    def __init__(self, section = None) -> None:
        self.kw_model = KeyBERT()
        # self.rt = RhymeTagger()
        # self.rt.load_model("en", verbose=False)

        if section != None:
            self.fill(section)

    def fill(self, section):
        section_list = section.strip().split(",")
        
        # lines count
        self.num_lines = len(section_list)

        # Keywords
        keywords = self.kw_model.extract_keywords(section)

        keywords_joined = ", ".join([x[0] for x in keywords])    
        url = 'http://lindat.mff.cuni.cz/services/translation/api/v2/models/en-cs'
        response = requests.post(url, data = {"input_text": keywords_joined})
        response.encoding='utf8'
        cs_keywords_joined = response.text[:-1]
        self.keywords = cs_keywords_joined.split(", ")

        # # rhyme scheme
        # rhymes = self.rt.tag(poem=section_list, output_format=3)
        # self.rhyme_scheme = self._fill_in_none_rhymes(rhymes)

        # syllables count
        self.syllables = [len(syllabify(sec)) for sec in section_list]


    def _fill_in_none_rhymes(self, rhymes):
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

