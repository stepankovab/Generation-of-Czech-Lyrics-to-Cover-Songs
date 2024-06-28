import re
from eval.syllabator import syllabify
import requests

class RhymeFinder:
    def __init__(self, lang = "cs") -> None:
        self.lang = lang

    def tag(self, poem, output_format = 0):
        if self.lang == "cs":
            return self._tag_cs(poem)
        elif self.lang == "en":
            return self._tag_en(poem)
        else:
            raise ValueError(f"Unsupported language: {self.lang}")

    def _tag_cs(self, lines):
        endings = []
        for line in lines:
            endings.append(self._get_rhyme_key_cs(line))
        
        rhyme_scheme = []
        endings_map = {}
        char_counter = 0

        for i in range(len(endings)):
            if endings[i] not in endings_map:
                # convert to capital letters, start with A -> chr(65)
                endings_map[endings[i]] = chr(65 + char_counter)
                char_counter += 1
            rhyme_scheme.append(endings_map[endings[i]])

        return rhyme_scheme                

    def _get_rhyme_key_cs(self, line : str):
        line = line.lower()
        line = re.sub("ch", "h", line)

        syll_line = syllabify(line)
        if len(syll_line) == 0:
            return "***"

        ending = "***" + syll_line[-1]
        ending = ending[-3:]

        replacements = [
            ("[sz]", "S"),
            ("[dt]", "D"),
            ("[gk]", "K"),
            ("[bp]", "B"),
            ("[vfw]", "V"),
            ("[ďť]", "Ď"),
            ("[aá]", "A"),
            ("[eé]", "E"),
            ("[žš]", "Š"),
            ("([Dn])([ií])", "ĎI"),
            ("[iíyý]", "I"),
            ("[oó]", "O"),
            ("[uúů]", "U"),
            ("^.[mn]ě", "*Ně"),
            ("^[AEIOU](.[AEIOU])", r"*\1"),
            ("^[čřĎ]", "Š"),
            ("^[ŠSDKBVrhjlcnm]([AEIOUě].)", r"*\1")     
        ]

        for (original, replacement) in replacements:
            ending = re.sub(original, replacement, ending)

        return ending

    def _tag_en(self, lines: list[str]):
        # find last words
        last_words = []
        for line in lines:
            if not line.strip():
                last = ""
            else:
                last = ''.join([x for x in re.split(r"\s", line.strip())[-1] if x.isalpha])
            last_words.append(last.lower())

        # fill in rhymes dict
        rhyming_words_dict = {}
        for ending in set(last_words):
            url = "https://rhymebrain.com/talk?function=getRhymes"
            response = requests.post(url, data = {"word": ending, "lang": self.lang})
            response.encoding='utf8'
            response_json = response.json()

            rhyming_words_dict[ending] = set([ending])
            for rhyme in response_json:
                rhyming_words_dict[ending].add(rhyme["t"])

        # find rhyming pairs
        rhyming_lines = {}
        for i in range(len(last_words)):
            rhyming_lines[i] = []
            for j in range(i + 1, len(last_words)):

                if last_words[i] == last_words[j]:
                    rhyming_lines[i].append(j)
                    continue

                if last_words[i] in rhyming_words_dict[last_words[j]]:
                    rhyming_lines[i].append(j)
                    continue

                if last_words[j] in rhyming_words_dict[last_words[i]]:
                    rhyming_lines[i].append(j)

        rhyme_scheme = [None for x in range(len(last_words))]
        char_counter = 0

        for i in range(len(last_words)):
            if rhyme_scheme[i] == None:
                # convert to capital letters, start with A -> chr(65)
                char_counter += 1
                rhyme_scheme[i] = chr(64 + char_counter)

            for j in rhyming_lines[i]:
                rhyme_scheme[j] = chr(64 + char_counter)
        
        return rhyme_scheme

