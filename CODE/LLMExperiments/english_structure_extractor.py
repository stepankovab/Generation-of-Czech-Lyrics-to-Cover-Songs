from eval.en_syllabator import syllabify
from eval.tagger import RhymeTagger
from eval.rhyme_finder import RhymeFinder
import requests
from keybert import KeyBERT


class SectionStructure:
    original_lyrics = []
    syllables = []
    rhyme_scheme = []
    keywords = []
    en_keywords = []
    line_keywords = []
    en_line_keywords = []
    num_lines: int

    def __init__(self, section = None, kw_model = KeyBERT(), rt = RhymeTagger()) -> None:
        self.kw_model = kw_model
        self.rt = rt

        if isinstance(self.rt, RhymeTagger):
            self.rt.load_model("en", verbose=False)

        if isinstance(self.rt, RhymeFinder):
            self.rt.lang = "en"

        if section != None:
            self.fill(section)

    def copy(self):
        """
        copy section structure as a new instance
        """
        new_section = SectionStructure(kw_model=self.kw_model, rt=self.rt)
        new_section.original_lyrics = self.original_lyrics.copy()
        new_section.syllables = self.syllables.copy()
        new_section.rhyme_scheme = self.rhyme_scheme.copy()
        new_section.keywords = self.keywords.copy()
        new_section.en_keywords = self.en_keywords.copy()
        new_section.line_keywords = self.line_keywords.copy()
        new_section.en_line_keywords = self.en_line_keywords.copy()
        new_section.num_lines = self.num_lines

        return new_section

    def fill(self, section):
        """
        section: divided by ','
        """
        self.reset()

        section_list = section.strip().split(",")
        self.original_lyrics = section_list
        
        # lines count
        self.num_lines = len(section_list)

        # Keywords
        keywords = self.kw_model.extract_keywords(section)
        self.en_keywords = [x[0] for x in keywords]

        keywords_joined = ", ".join(self.en_keywords)    
        url = 'http://lindat.mff.cuni.cz/services/translation/api/v2/models/en-cs'
        response = requests.post(url, data = {"input_text": keywords_joined})
        response.encoding='utf8'
        cs_keywords_joined = response.text[:-1]
        self.keywords = cs_keywords_joined.split(", ")

        # Line keywords
        for i in range(len(section_list)):
            keywords = self.kw_model.extract_keywords(section_list[i])
            if keywords == []:
                self.line_keywords.append("")
                self.en_line_keywords.append("")
                continue
            self.en_line_keywords.append(' '.join([x[0] for x in keywords[:min(len(keywords), 2)]]))
            response = requests.post(url, data = {"input_text": self.en_line_keywords[-1]})
            response.encoding='utf8'
            cs_keywords_line = response.text[:-1]
            self.line_keywords.append(cs_keywords_line)

        # rhyme scheme
        self.rhyme_scheme = self.rt.tag(poem=section_list, output_format=3)
        if isinstance(self.rt, RhymeTagger):
            self.rhyme_scheme = self._fill_in_none_rhymes(self.rhyme_scheme)

        # syllables count
        self.syllables = [len(syllabify(sec)) for sec in section_list]

    def reset(self):
        self.original_lyrics = []
        self.syllables = []
        self.rhyme_scheme = []
        self.keywords = []
        self.en_keywords = []
        self.line_keywords = []
        self.en_line_keywords = []
        self.num_lines = 0


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

