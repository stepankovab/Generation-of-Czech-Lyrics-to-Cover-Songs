from eval.syllabator import syllabify
import requests
from keybert import KeyBERT
from sentence_transformers import SentenceTransformer, util
from english_structure_extractor import SectionStructure
import re
from eval.tagger import RhymeTagger


class Evaluator():

    def __init__(self) -> None:
        pass
        self.kw_model = KeyBERT()
        self.embed_model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
        self.rt_cs = RhymeTagger()
        self.rt_cs.load_model("cs", verbose=False)

    def evaluate_outputs_structure(self, outputs_w_structures: list[tuple[str, SectionStructure]]):
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
                        "semantic_sim" : [],
                        "keyword_sim" : [],
                        "line_keyword_sim" : []}


        """
        results_dict = {"syll_dist" : [],
                        "syll_acc" : [],
                        "rhyme_scheme_agree" : [],
                        "semantic_sim" : [],
                        "keyword_sim" : [],
                        "line_keyword_sim" : []}

        for output, structure in outputs_w_structures:
            output = output.split(",")

            assert len(output) == structure.num_lines
            

            out_syllables = []
            out_endings = []

            for line_i in range(len(output)):
                line = output[line_i]

                if not line.strip():
                    out_syllables.append(0)
                    out_endings.append("")
                    continue

                line = line.split("#")[-1].strip()

                if not line:
                    out_syllables.append(0)
                    out_endings.append("")
                    continue

                syllabified_line = syllabify(line)
                out_syllables.append(len(syllabified_line))
                out_endings.append(syllabified_line[-1][-min(len(syllabified_line[-1]), 3):])

            ##################### metrics ####################

            # syllable distance
            results_dict["syll_dist"].append(self.get_section_syllable_distance(structure.syllables, out_syllables))

            # syllable accuracy
            syll_accuracy = 0
            positive = 0
            for i in range(len(out_syllables)):
                if out_syllables[i] == structure.syllables[i]:
                    positive += 1
            if len(out_syllables) > 0:
                syll_accuracy = positive / len(out_syllables)
            results_dict["syll_acc"].append(syll_accuracy)

            # rhyme scheme distance
            rhymes_cs = self.rt_cs.tag(poem=output, output_format=3)
            cs_rhyme_scheme = self._fill_in_none_rhymes(rhymes_cs)

            results_dict["rhyme_scheme_agree"].append(self.get_rhyme_scheme_agreement(structure.rhyme_scheme, cs_rhyme_scheme))

            # semantic similarity
            semantic_similarity = 0
            if len(structure.original_lyrics) > 0:
                url = 'http://lindat.mff.cuni.cz/services/translation/api/v2/models/cs-en'
                response = requests.post(url, data = {"input_text": ' '.join(output)})
                response.encoding='utf8'
                translated_output = response.text[:-1]
                semantic_similarity = self.get_semantic_similarity(translated_output, ' '.join(structure.original_lyrics))
            results_dict["semantic_sim"].append(semantic_similarity)

            # keyword similarity
            if len(structure.en_keywords) > 0:
                keyword_similarity = self.get_keyword_semantic_similarity(structure.en_keywords, output, keywords_in_en=True, output_in_en = False)
            else:
                keyword_similarity = self.get_keyword_semantic_similarity(structure.keywords, output, keywords_in_en=False, output_in_en = False)
            results_dict["keyword_sim"].append(keyword_similarity)

            # line keyword similarity
            if len(structure.en_line_keywords) == structure.num_lines:
                line_keywords_similarity = self.get_line_keyword_semantic_similarity(structure.en_line_keywords, output, keywords_in_en=True, output_in_en = False)
            else:
                line_keywords_similarity = self.get_line_keyword_semantic_similarity(structure.line_keywords, output, keywords_in_en=False, output_in_en = False)
            results_dict["line_keyword_sim"].append(line_keywords_similarity)

        return results_dict

    def get_line_keyword_semantic_similarity(self, keywords, output, keywords_in_en = True, output_in_en = True):
        """
        returns average similarity of a line to keywords
        
        """
        if not keywords_in_en:
            # Keywords to english
            for i in range(len(keywords)):
                url = 'http://lindat.mff.cuni.cz/services/translation/api/v2/models/cs-en'
                response = requests.post(url, data = {"input_text": keywords[i]})
                response.encoding='utf8'
                keywords[i] = response.text[:-1]

        if not output_in_en:
            # output to english
            url = 'http://lindat.mff.cuni.cz/services/translation/api/v2/models/cs-en'
            for i in range(len(output)):
                response = requests.post(url, data = {"input_text": output[i]})
                response.encoding='utf8'
                output[i] = response.text[:-1]

        # Extract new keywords
        out_keywords = []
        for i in range(len(output)):
            line_keywords = [x[0] for x in self.kw_model.extract_keywords(output[i])]
            out_keywords.append(' '.join(line_keywords[:min(len(line_keywords), 2)]))

        print(keywords)
        print(out_keywords)
        print()

        similarities_per_line = []
        for i in range(len(keywords)):
            similarities_per_line.append(self.get_semantic_similarity(keywords[i], out_keywords[i]))

        return sum(similarities_per_line) / len(similarities_per_line)


    def get_keyword_semantic_similarity(self, keywords, output, keywords_in_en = True, output_in_en = True):
        if not keywords_in_en:
            # Keywords to english
            keywords_joined = ", ".join(keywords)    
            url = 'http://lindat.mff.cuni.cz/services/translation/api/v2/models/cs-en'
            response = requests.post(url, data = {"input_text": keywords_joined})
            response.encoding='utf8'
            en_keywords_joined = response.text[:-1]
            keywords = en_keywords_joined.split(", ")

        if not output_in_en:
            # output to english
            lyrics_joined = ", ".join(output)    
            url = 'http://lindat.mff.cuni.cz/services/translation/api/v2/models/cs-en'
            response = requests.post(url, data = {"input_text": lyrics_joined})
            response.encoding='utf8'
            output = response.text[:-1]

        # Extract new keywords
        out_keywords = [x[0] for x in self.kw_model.extract_keywords(output)]
        if out_keywords == []:
            out_keywords = [""]

        print(keywords)
        print(out_keywords)
        print()

        return self.get_semantic_similarity(out_keywords, keywords)


    def get_semantic_similarity(self, text1, text2):
        """
        Embed two texts and get their cosine similarity
        """
        embedding1 = self.embed_model.encode(text1, convert_to_tensor=False)
        embedding2 = self.embed_model.encode(text2, convert_to_tensor=False)
        
        cosine_similarity = util.cos_sim(embedding1, embedding2)

        return cosine_similarity[0][0].item()

    def get_section_syllable_distance(self, syllables, out_syllables):
        distance = 0
        
        assert len(syllables) == len(out_syllables)

        for i in range(len(out_syllables)):
            distance += (abs(syllables[i] - out_syllables[i]) / max(syllables[i], 1)) + (abs(syllables[i] - out_syllables[i]) / max(out_syllables[i], 1))
        distance /= (2 * len(out_syllables))
        return distance
    

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
        assert len(desired_scheme) == len(new_scheme)
        assert len(desired_scheme) > 0

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









































    # def eval_from_first_line(self, model_output):
    #     """
    #     Evaluate model outputs based on syllables, rhymes and semantics criteria read from first line

    #     Parameters
    #     --------------
    #     model_output: The output of a model

    #     Returns
    #     --------------
    #     results
    #     """
    #     model_output = model_output.split("\n")

    #     eval_info = [x.strip() for x in model_output[0].split("#") if x.strip()]
    #     if len(eval_info) == 0:
    #         return None, None, None, None
    #     if len(eval_info) > 0:
    #         syllables = [int(x) for x in eval_info[0].split(" ")]
    #     if len(eval_info) > 1:
    #         keywords = eval_info[1].split(" ")
    #     if len(eval_info) > 2:
    #         endings = eval_info[2].split(" ")

    #     expected_length = len(syllables)
    #     length_ratio = (len(model_output) - 1) / expected_length

    #     out_syllables = []
    #     out_endings = []
    #     out_lines = []

    #     for line_i in range(1, min(expected_length + 1,len(model_output))):
    #         line = model_output[line_i]

    #         if not line.strip():
    #             continue

    #         line = line.split("#")[-1].strip()

    #         if not line:
    #             continue

    #         out_lines.append(line)

    #         syllabified_line = syllabify(line)

    #         out_syllables.append(len(syllabified_line))
    #         # out_endings.append(syllabified_line[-1])

    #     # syllable distance
    #     syll_distance = None
    #     if len(syllables) > 0:
    #         syll_distance = self.get_section_syllable_distance(syllables, out_syllables)

    #     # syllable accuracy
    #     syll_accuracy = None
    #     if len(syllables) > 0:
    #         positive = 0
    #         for i in range(len(out_syllables)):
    #             if out_syllables[i] == syllables[i]:
    #                 positive += 1
    #         syll_accuracy = positive / len(out_syllables)

    #     # line endings accuracy
    #     end_accuracy = None
    #     if len(endings) > 0:
    #         positive = 0
    #         for i in range(len(out_endings)):
    #             if out_endings[i] == endings[i]:
    #                 positive += 1
    #         end_accuracy = positive / len(out_endings)

    #     # keyword similarity
    #     keyword_similarity = None
    #     if len(keywords) > 0:
    #         keyword_similarity, out_keywords = self.get_keyword_semantic_similarity(keywords, out_lines)

    #     print(syllables)
    #     print(out_syllables)
    #     print()
    #     print(endings)
    #     print(out_endings)
    #     print()

    #     return length_ratio, syll_distance, syll_accuracy, end_accuracy, keyword_similarity








# evaluator = Evaluator()



# output = ["4 6 4 8 # čas půlnoc komáři kostel kytka #","6 # co se v kostele děje","4 # co se v kostele děje","4 # co se v kostele děje","8 # co z kostela plyne k ránu","8 # co se v kostele dít má","8 # co z kostela plyne k ránu do noci","6 # co z kostela vychází k ránu","8 # co z kostela odchází k ránu do noci","4 # co z kostela plyne k ránu"]


# evaluator.eval_output(output)
