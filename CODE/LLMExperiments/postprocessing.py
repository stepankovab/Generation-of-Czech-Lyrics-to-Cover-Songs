from evaluator import Evaluator
from english_structure_extractor import SectionStructure
from eval.syllabator import syllabify


class Postprocesser():
    def __init__(self, evaluator = Evaluator()) -> None:
        self.evaluator = evaluator

    def choose_best_line(self, lines, syllables_in = None, ending_in = None, keywords_in = None, english_in = None, czech_in = None):
        scores = [[] for _ in range(len(lines))]
        for line_i in range(len(lines)):
            sylls = syllabify(lines[line_i])
            if len(sylls) == 0:
                continue

            if syllables_in != None:
                syllables_out = len(sylls)
                scores[line_i].append(abs(syllables_in - syllables_out))
            
            if ending_in != None:
                ending_out = sylls[-1]
                scheme = self.evaluator.rt.tag([ending_in, ending_out])
                if scheme[0] == scheme[1] and scheme[0] != None:
                    scores[line_i].append(0)
                else:
                    scores[line_i].append(1)

            similarity = 0
            measurings = 0
            if keywords_in != None:
                similarity += self.evaluator.get_keyword_semantic_similarity([keywords_in], lines[line_i], keywords_in_en=False, output_in_en=False)
                measurings += 1
            if english_in != None:
                similarity += self.evaluator.get_semantic_similarity(lines[line_i], english_in, text1_in_en=False)
                measurings += 1
            if czech_in != None:
                similarity += self.evaluator.get_semantic_similarity(lines[line_i], czech_in, text1_in_en=False, text2_in_en=False)
                measurings += 1
            if measurings != 0:
                # smaller is better
                scores[line_i].append(1 - similarity / measurings)

        for s in scores:
            # TODO najit nejaky co dominuje vsem, pokud ne tak po slozkach a nejak vyvazovat...
            pass


    def choose_best_section(self):
        pass

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
pc.choose_best_line(["tohle je test", "je to zkouska", "tohle neni trest"], 4, "est", "test", "It is a test", "Je to test")