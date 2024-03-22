import re
from eval.syllabator import syllabify
from HT_loader import HT_loader


class RhymeFinder:
    def __init__(self, lang = "cs") -> None:
        self.lang = lang

    def tag(self, lines):
        if self.lang == "cs":
            return self._tag_cs(lines)
        elif self.lang == "en":
            return self._tag_en(lines)
        else:
            raise ValueError(f"Unsupported language: {self.lang}")


    def _tag_cs(self, lines):
        endings = []
        for line in lines:
            endings.append(self.get_rhyme_key_cs(line))
        
        rhyme_scheme = [None for x in range(len(endings))]
        endings_map = {}
        char_counter = 0

        for i in range(len(endings)):
            if endings[i] not in endings_map:
                # convert to capital letters, start with A -> chr(65)
                endings_map[endings[i]] = chr(65 + char_counter)
                char_counter += 1
            rhyme_scheme[i] = endings_map[endings[i]]

        print(rhyme_scheme)

        return rhyme_scheme                


    def _get_rhyme_key_cs(self, line : str):
        line = line.lower()
        line = re.sub("ch", "h", line)

        ending = "***" + syllabify(line)[-1]
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
            ("([Dn])([ií])", "ĎI"),
            ("[iíyý]", "I"),
            ("[oó]", "O"),
            ("[uúů]", "U"),
            ("^.[mn]ě", "*Ně"),
            ("^[AEIOU](.[AEIOU])", r"*\1"),
            ("^[ščžřĎ]", "Š"),
            ("^[ŠSDKBVrhjlcnm]([AEIOUě].)", r"*\1")     
        ]

        for (original, replacement) in replacements:
            ending = re.sub(original, replacement, ending)

        print(ending)
        return ending



    def _tag_en(self, lines):
        pass


    def _get_rhyme_key_cs(self, line):
        pass

