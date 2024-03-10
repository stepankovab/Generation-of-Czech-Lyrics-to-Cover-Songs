from eval.syllabator import syllabify
import requests
from keybert import KeyBERT
from sentence_transformers import SentenceTransformer, util


class Evaluator():

    def __init__(self) -> None:
        self.kw_model = KeyBERT()
        self.embed_model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')

    def eval_output(self, model_output, read_output = True, syllables = [], endings = [], keywords = []):
        """
        Evaluate model outputs based on syllables, rhymes and semantics criteria.

        Parameters
        --------------
        model_output: The output of a model
        read_output: bool, if eval should be based on the first line of the model
        syllables: The expected syllables counts for each line
        endings: The expected line endings for each line
        keywords: The expected keywords of the output

        Returns
        --------------
        results
        """
        model_output = model_output.split("\n")

        if read_output == True:
            eval_info = [x.strip() for x in model_output[0].split("#") if x.strip()]
            if len(eval_info) == 0:
                return None, None, None, None
            if len(eval_info) > 0:
                syllables = [int(x) for x in eval_info[0].split(" ")]
            if len(eval_info) > 1:
                keywords = eval_info[1].split(" ")
            if len(eval_info) > 2:
                endings = eval_info[2].split(" ")

        expected_length = len(syllables)
        length_ratio = (len(model_output) - 1) / expected_length

        out_syllables = []
        out_endings = []
        out_lines = []

        for line_i in range(1, min(expected_length + 1,len(model_output))):
            line = model_output[line_i]

            if not line.strip():
                continue

            line = line.split("#")[-1].strip()

            if not line:
                continue

            out_lines.append(line)

            syllabified_line = syllabify(line)

            out_syllables.append(len(syllabified_line))
            # out_endings.append(syllabified_line[-1])

        # syllable distance
        syll_distance = None
        if len(syllables) > 0:
            syll_distance = self.get_section_syllable_distance(syllables, out_syllables)

        # syllable accuracy
        syll_accuracy = None
        if len(syllables) > 0:
            positive = 0
            for i in range(len(out_syllables)):
                if out_syllables[i] == syllables[i]:
                    positive += 1
            syll_accuracy = positive / len(out_syllables)

        # line endings accuracy
        end_accuracy = None
        if len(endings) > 0:
            positive = 0
            for i in range(len(out_endings)):
                if out_endings[i] == endings[i]:
                    positive += 1
            end_accuracy = positive / len(out_endings)

        # keyword similarity
        keyword_similarity = None
        if len(keywords) > 0:
            keyword_similarity, out_keywords = self.get_keyword_semantic_similarity(keywords, out_lines)

        print(syllables)
        print(out_syllables)
        print()
        print(endings)
        print(out_endings)
        print()

        return length_ratio, syll_distance, syll_accuracy, end_accuracy, keyword_similarity




    def get_keyword_semantic_similarity(self, keywords, model_output):
        # Keywords to english
        keywords_joined = ", ".join(keywords)    
        url = 'http://lindat.mff.cuni.cz/services/translation/api/v2/models/cs-en'
        response = requests.post(url, data = {"input_text": keywords_joined})
        response.encoding='utf8'
        en_keywords_joined = response.text[:-1]
        keywords = en_keywords_joined.split(", ")

        # Extract new keywords
        lyrics_joined = ", ".join(model_output)    
        url = 'http://lindat.mff.cuni.cz/services/translation/api/v2/models/cs-en'
        response = requests.post(url, data = {"input_text": lyrics_joined})
        response.encoding='utf8'
        en_lyrics_joined = response.text[:-1]

        out_keywords = [x[0] for x in self.kw_model.extract_keywords(en_lyrics_joined)]

        print(keywords)
        print(out_keywords)
        print()

        
        embedding1 = self.embed_model.encode(keywords, convert_to_tensor=False)
        if out_keywords != []:
            embedding2 = self.embed_model.encode(out_keywords, convert_to_tensor=False)
        else:
            embedding2 = self.embed_model.encode(model_output, convert_to_tensor=False)
        
        cosine_similarity = util.cos_sim(embedding1, embedding2)

        return(cosine_similarity[0][0].item(), out_keywords)

    def get_section_syllable_distance(self, syllables, out_syllables):
        distance = 0
        for i in range(len(out_syllables)):
            distance += (abs(syllables[i] - out_syllables[i]) / max(syllables[i], 1)) + (abs(syllables[i] - out_syllables[i]) / max(out_syllables[i], 1))
        distance /= (2 * len(out_syllables))
        return distance
    



# evaluator = Evaluator()

# output = ["4 6 4 8 # čas půlnoc komáři kostel kytka #","6 # co se v kostele děje","4 # co se v kostele děje","4 # co se v kostele děje","8 # co z kostela plyne k ránu","8 # co se v kostele dít má","8 # co z kostela plyne k ránu do noci","6 # co z kostela vychází k ránu","8 # co z kostela odchází k ránu do noci","4 # co z kostela plyne k ránu"]


# evaluator.eval_output(output)
