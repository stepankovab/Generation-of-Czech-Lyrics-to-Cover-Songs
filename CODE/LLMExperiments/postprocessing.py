from evaluator import Evaluator
from english_structure_extractor import SectionStructure
from eval.syllabator import syllabify


class Postprocesser():
    def __init__(self, evaluator = Evaluator()) -> None:
        self.evaluator = evaluator

    def choose_best_line(self, lines_list, syll_dist_tollerance = 0.2, syllables_in = None, ending_in = None, keywords_in = None, text_in = None, text_in_english = True):
        if len(lines_list) == 0:
            return ""
        if syllables_in == None and ending_in == None and keywords_in == None and text_in == None:
            return lines_list[0]
            
        scores_dict = {"syllables" : [], "endings" : [], "similarity" : []}
        scores = [0 for _ in range(len(lines_list))]

        for line_i in range(len(lines_list)):
            sylls = syllabify(lines_list[line_i])
            
            # fill in scores dict
            if syllables_in != None:
                syllables_out = len(sylls)
                scores_dict["syllables"].append(self.evaluator.get_section_syllable_distance([syllables_in], [syllables_out]))
            
            if ending_in != None:
                if len(sylls) == 0:
                    scores_dict["endings"].append(0)
                    continue
                ending_out = sylls[-1]
                scheme = self.evaluator.rt.tag([ending_in, ending_out])
                if scheme[0] == scheme[1] and scheme[0] != None:
                    scores_dict["endings"].append(0)
                else:
                    scores_dict["endings"].append(1)

            similarity = 0
            measurings = 0
            if keywords_in != None:
                similarity += self.evaluator.get_keyword_semantic_similarity([keywords_in], [lines_list[line_i]], keywords_in_en=False, output_in_en=False)
                measurings += 1
            if text_in != None:
                similarity += self.evaluator.get_semantic_similarity(lines_list[line_i], text_in, text1_in_en=False, text2_in_en=text_in_english)
                measurings += 1
            if measurings != 0:
                scores_dict["similarity"].append(similarity / measurings)

        # pick the best match
        syll_multip_factor = 10 / max(syll_dist_tollerance, 0.0001)
        for line_i in range(len(lines_list)):
            if syllables_in != None:
                scores[line_i] += syll_multip_factor * scores_dict["syllables"][line_i]
            if ending_in != None:
                scores[line_i] += 10 * scores_dict["endings"][line_i]
            if keywords_in != None or text_in != None:
                scores[line_i] += (1 - scores_dict["similarity"][line_i])

        ordered_indicies = [i for i, x in sorted(enumerate(scores), key=lambda x: x[1])]

        for i in ordered_indicies:
            print(lines_list[i])

        return lines_list[ordered_indicies[0]]

    def choose_best_section(self, lyrics_list, structure : SectionStructure, syll_dist_tollerance = 0.2, rhyme_scheme_agree_tollerance = 0.5):
        scores_dict = self.evaluator.evaluate_outputs_structure([(','.join(lyrics), structure) for lyrics in lyrics_list])
        scores = [0 for _ in range(len(lyrics_list))]

        # pick the best match
        syll_multip_factor = 10 / max(syll_dist_tollerance, 0.0001)
        rhyme_multip_factor = 10 / (1 - min(rhyme_scheme_agree_tollerance, 0.9999))
        for line_i in range(len(lyrics_list)):
            scores[line_i] += syll_multip_factor * scores_dict["syll_dist"][line_i]
            scores[line_i] += rhyme_multip_factor * scores_dict["rhyme_scheme_agree"][line_i]
            scores[line_i] += (1 - ((scores_dict["semantic_sim"][line_i] + 
                                     scores_dict["keyword_sim"][line_i] + 
                                     scores_dict["line_keyword_sim"][line_i])/3))

        ordered_indicies = [i for i, x in sorted(enumerate(scores), key=lambda x: x[1])]

        for i in ordered_indicies:
            print(lyrics_list[i])
            print()

        return lyrics_list[ordered_indicies[0]]

    def correct_length_word(self):
        pass

    def correct_length_by_rhyme(self):
        pass

    def correct_length_remove_add_stopwords(self):
        pass

    def choose_best_subsection(self):
        pass

    def put_together_best_subsection(self):
        pass







pc = Postprocesser()
pc.choose_best_line(["tohle je test", "je to zkouska", "tohle neni trest"], syllables_in=4, ending_in="est", keywords_in="test", text_in="Je to test", text_in_english=False)

