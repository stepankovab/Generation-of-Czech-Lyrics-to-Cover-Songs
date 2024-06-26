from eval.syllabator import syllabify
from keybert import KeyBERT
from sentence_transformers import SentenceTransformer, util
from english_structure_extractor import SectionStructure
import re
from rhymetagger import RhymeTagger
from eval.rhyme_finder import RhymeFinder
from eval.same_word_tagger import SameWordRhymeTagger
from rhymer_types import RhymerType
from eval.phon_czech import ipa_czech
import eng_to_ipa as ipa
from eval.chrF import computeChrF
from nltk.translate.bleu_score import sentence_bleu
from translate import lindat_translate

class Evaluator():

    def __init__(self, czech_rhyme_detector = RhymeFinder(), kw_model = KeyBERT(), embed_model = SentenceTransformer('all-MiniLM-L6-v2'), verbose = False) -> None:
        """
        Parameters
        ---------------
        rt: "cs" RhymeTagger, RhymeFinder, or any rhymer with 'tag(self, poem : list[str], output_format : int)' method
        kw_model: keyword finder
        embed_model: sentence embedding model
        """
        
        self.kw_model = kw_model
        self.embed_model = embed_model

        if isinstance(czech_rhyme_detector, RhymerType):
            if czech_rhyme_detector == RhymerType.RHYMETAGGER:
                czech_rhyme_detector = RhymeTagger()
            elif czech_rhyme_detector == RhymerType.RHYMEFINDER:
                czech_rhyme_detector = RhymeFinder()
            elif czech_rhyme_detector == RhymerType.SAME_WORD_RHYMETAGGER:
                czech_rhyme_detector = SameWordRhymeTagger("cs")

        self.czech_rhyme_detector = czech_rhyme_detector
        self.verbose = verbose

        if isinstance(self.czech_rhyme_detector, RhymeTagger):
            self.czech_rhyme_detector.load_model("cs", verbose=False)

        if isinstance(self.czech_rhyme_detector, RhymeFinder):
            self.czech_rhyme_detector.lang = "cs"

        if isinstance(self.czech_rhyme_detector, SameWordRhymeTagger):
            self.czech_rhyme_detector.load_model("cs")

    def evaluate_outputs_structure(self, outputs_w_structures: list[tuple[str, SectionStructure]], evaluate_keywords=False, evaluate_line_keywords=False, evaluate_translations = False):
        """
        outputs_w_structures: list(str, SectionStructure)
        
        syllables: The expected syllables counts for each line
        endings: The expected line endings for each line
        keywords: The expected keywords of the output
        
        Returns
        ---------------
        results_dict = {"syll_dist" : [],
                        "syll_acc" : [],
                        "rhyme_scheme_agree" : [],
                        "rhyme_accuracy" : [],
                        "semantic_sim" : [],
                        "keyword_sim" : [],
                        "line_keyword_sim" : [],
                        "phon_rep_dif" : [],
                        "bleu2gram" : [],
                        "chrf" : []}
        """
        results_dict = {"syll_dist" : [],
                        "syll_acc" : [],
                        "rhyme_scheme_agree" : [],
                        "rhyme_accuracy" : [],
                        "semantic_sim" : [],
                        "keyword_sim" : [],
                        "line_keyword_sim" : [],
                        "phon_rep_dif" : [],
                        "bleu2gram" : [],
                        "chrf" : []}

        for output, structure in outputs_w_structures:
            output_list = output.split(",")         
            out_syllables = []

            for line_i in range(len(output_list)):
                line = output_list[line_i]

                if not line.strip():
                    out_syllables.append(0)
                    continue

                syllabified_line = syllabify(line)

                if len(syllabified_line) == 0:
                    out_syllables.append(0)
                    continue

                out_syllables.append(len(syllabified_line))

            ##################### metrics ####################

            # syllable distance
            results_dict["syll_dist"].append(self.get_section_syllable_distance(structure.syllables, out_syllables))

            # syllable accuracy
            syll_accuracy = 0
            positive = 0
            if len(structure.syllables) == len(out_syllables) and len(out_syllables) > 0:
                for i in range(len(out_syllables)):
                    if out_syllables[i] == structure.syllables[i]:
                        positive += 1
                syll_accuracy = positive / len(out_syllables)
            results_dict["syll_acc"].append(syll_accuracy)

            # rhyme scheme agreement
            output_rhyme_scheme = self.czech_rhyme_detector.tag(poem=output_list, output_format=3)
            if isinstance(self.czech_rhyme_detector, RhymeTagger):
                output_rhyme_scheme = self._fill_in_none_rhymes(output_rhyme_scheme)

            results_dict["rhyme_scheme_agree"].append(self.get_rhyme_scheme_agreement(structure.rhyme_scheme, output_rhyme_scheme))

            # rhyme scheme accuracy
            results_dict["rhyme_accuracy"].append(self.get_rhyme_scheme_accuracy(structure.rhyme_scheme, output_rhyme_scheme))

            # semantic similarity
            semantic_similarity = 0
            if len(structure.original_lyrics_list) > 0:
                semantic_similarity = self.get_semantic_similarity(output_list, structure.original_lyrics_list, text1_in_en=False)
            results_dict["semantic_sim"].append(semantic_similarity)

            # keyword similarity
            keyword_similarity = 0
            if evaluate_keywords == True:
                if len(structure.en_keywords) > 0:
                    keyword_similarity = self.get_keyword_semantic_similarity(structure.en_keywords, output_list, keywords_in_en=True, output_in_en = False)
                else:
                    keyword_similarity = self.get_keyword_semantic_similarity(structure.keywords, output_list, keywords_in_en=False, output_in_en = False)
            results_dict["keyword_sim"].append(keyword_similarity)

            # line keyword similarity
            line_keywords_similarity = 0
            if evaluate_line_keywords == True:
                if len(structure.en_line_keywords) == structure.num_lines:
                    line_keywords_similarity = self.get_line_keyword_semantic_similarity(structure.en_line_keywords, output_list, keywords_in_en=True, output_in_en = False)
                else:
                    line_keywords_similarity = self.get_line_keyword_semantic_similarity(structure.line_keywords, output_list, keywords_in_en=False, output_in_en = False)
            results_dict["line_keyword_sim"].append(line_keywords_similarity)

            # phoneme repetition difference
            cs_dist2 = self.get_phoneme_distinct2(output_list, "cs")
            en_dist2 = self.get_phoneme_distinct2(structure.original_lyrics_list, "en")
            results_dict["phon_rep_dif"].append(abs(en_dist2 - cs_dist2))

            # bleu score and chrf score
            bleu2gram = 0
            totalF = 0
            if evaluate_translations == True:
                cs_lyrics = lindat_translate(structure.original_lyrics_list, "en", "cs", ", ")
                reference = [cs_lyrics.replace(',', '').split()]
                candidate = ' '.join(output_list).split()
                bleu2gram = sentence_bleu(reference, candidate, weights=(0.5, 0.5))
                try:
                    totalF, _, _, _ = computeChrF([cs_lyrics], [' '.join(output_list)], nworder=2, ncorder=6, beta=2)
                except:
                    pass
            results_dict["chrf"].append(totalF)
            results_dict["bleu2gram"].append(bleu2gram)
        return results_dict
    
    def get_line_keyword_semantic_similarity(self, keywords, output_lines, keywords_in_en = True, output_in_en = True):
        """
        returns average similarity of a line to keywords
        
        """
        translated_keywords = keywords.copy()
        if not keywords_in_en:
            # Keywords to english
            for i in range(len(keywords)):
                if not keywords[i].strip():
                    continue
                translated_keywords[i] = lindat_translate([keywords[i]], "cs", "en", " ")

        translated_output_lines = output_lines.copy()
        if not output_in_en:
            # output to english
            for i in range(len(translated_output_lines)):
                if not translated_output_lines[i].strip():
                    continue
                translated_output_lines[i] = lindat_translate([translated_output_lines[i]], "cs", "en", " ")

        # Extract new keywords
        out_keywords = []
        for i in range(len(translated_output_lines)):
            line_keywords = [x[0] for x in self.kw_model.extract_keywords(translated_output_lines[i])]
            out_keywords.append(' '.join(line_keywords[:min(len(line_keywords), 2)]))

        if len(out_keywords) != len(translated_keywords):
            return 0

        similarities_per_line = []
        for i in range(len(translated_keywords)):
            similarities_per_line.append(self.get_semantic_similarity(translated_keywords[i], out_keywords[i]))

        return sum(similarities_per_line) / len(similarities_per_line)


    def get_keyword_semantic_similarity(self, keywords, output_lines, keywords_in_en = True, output_in_en = True):
        translated_keywords = keywords.copy()
        if not keywords_in_en and ', '.join(translated_keywords).strip():
            translated_keywords = lindat_translate(translated_keywords, "cs", "en", ", ")

        translated_output_lines = output_lines.copy()
        if not output_in_en and ', '.join(translated_output_lines).strip():
            translated_output_lines = lindat_translate(translated_output_lines, "cs", "en", ", ")

        # Extract new keywords
        out_keywords = ", ".join([x[0] for x in self.kw_model.extract_keywords(translated_output_lines)])
            
        return self.get_semantic_similarity(out_keywords, translated_keywords)


    def get_semantic_similarity(self, text1, text2, text1_in_en = True, text2_in_en = True):
        """
        Embed two texts and get their cosine similarity
        """
        if not isinstance(text1, str):
            text1 = ', '.join(text1)
        if not isinstance(text2, str):
            text2 = ', '.join(text2)

        if not text1.strip():
            return 0
        if not text2.strip():
            return 0

        if not text1_in_en and ' '.join(text1).strip():
            text1 = lindat_translate([text1], "cs", "en", " ")

        if not text2_in_en and ' '.join(text2).strip():
            text2 = lindat_translate([text2], "cs", "en", " ")

        embedding1 = self.embed_model.encode(text1, convert_to_tensor=False)
        embedding2 = self.embed_model.encode(text2, convert_to_tensor=False)
        
        cosine_similarity = util.cos_sim(embedding1, embedding2)

        return cosine_similarity[0][0].item()

    def get_section_syllable_distance(self, syllables, out_syllables):
        distance = 0
        
        if len(syllables) != len(out_syllables):
            return 10

        for i in range(len(out_syllables)):
            distance += (abs(syllables[i] - out_syllables[i]) / max(syllables[i], 1)) + (abs(syllables[i] - out_syllables[i]) / max(out_syllables[i], 1))
        distance /= (2 * len(out_syllables))
        return distance
    

    def get_phoneme_distinct2(self, section : list[str], language : str) -> float:
        """
        language: the language of the section ['cs', 'en']
        """
        bigram_dict = {}
        bigram_count = 0

        for i in range(len(section)):
            if language == "cs":
                phonemes = ipa_czech(section[i]).split() + ["^"]
            elif language == "en":
                phonemes = re.sub('[ˈ\s]', '', ipa.convert(section[i], stress_marks='primary')) + "^"

            for p in range(len(phonemes) - 1):
                bigram = phonemes[p] + phonemes[p + 1]
                bigram_count += 1

                if bigram not in bigram_dict:
                    bigram_dict[bigram] = 0
                
                bigram_dict[bigram] += 1

        return len(bigram_dict)/ max(bigram_count,1)
 

    def get_rhyme_scheme_agreement(self, desired_scheme, new_scheme, alpha = 0.5):
        """
        Get agreement between rhyme schemes.

        Computed as:
            alpha * (rhyme agreement / num desired rhymes) + (1 - alpha) * (min(rhymes in either scheme) / num desired rhymes)

        Parameters
        --------------
        desired_scheme: list, The desired rhyme scheme (eg. [1,2,1,2])
        new_scheme: list, The new rhyme scheme to test against the desired scheme (eg. [1,1,2,2])
        alpha: int, koeficient for balancing importance of rhyme agreement and rhyme count, defaults to 0.5

        Returns
        --------------
        Agreement between rhyme schemes: int, [0,1], 1 is the best
        """
        try:
            assert len(desired_scheme) == len(new_scheme)
            assert len(desired_scheme) > 0
        except:
            return 0

        desired_edges = set()
        new_edges = set()

        for i in range(len(desired_scheme)):
            for j in range(i + 1, len(desired_scheme)):
                if desired_scheme[i] == desired_scheme[j]:
                    desired_edges.add((i,j))
                if new_scheme[i] == new_scheme[j]:
                    new_edges.add((i,j))

        edge_agreement = desired_edges.intersection(new_edges)

        if len(desired_edges) == 0:
            return 1

        return (alpha * (len(edge_agreement) / len(desired_edges))) + ((1 - alpha) * (min(len(desired_edges), len(new_edges)) / len(desired_edges)))

    def get_rhyme_scheme_accuracy(self, desired_scheme, new_scheme):
        try:
            assert len(desired_scheme) == len(new_scheme)
            assert len(desired_scheme) > 0
        except:
            return 0

        desired_edges = set()
        new_edges = set()

        for i in range(len(desired_scheme)):
            for j in range(i + 1, len(desired_scheme)):
                if desired_scheme[i] == desired_scheme[j]:
                    desired_edges.add((i,j))
                if new_scheme[i] == new_scheme[j]:
                    new_edges.add((i,j))

        same_edges = desired_edges.intersection(new_edges)
        all_edges = desired_edges.union(new_edges)

        if len(all_edges) == 0:
            return 1

        return len(same_edges) / len(all_edges)
    
    
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

