import requests
from sentence_transformers import SentenceTransformer, util


def cosine_section_similarity(czech_lines : list[str], english_lines : list[str]) -> float:

    czech_joined = ", ".join(czech_lines)    
    english_joined = ", ".join(english_lines)

    url = 'http://lindat.mff.cuni.cz/services/translation/api/v2/models/cs-en'
    response = requests.post(url, data = {"input_text": czech_joined})
    response.encoding='utf8'
    cz_to_english_joined = response.text

    model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')  # multi-language model

    embedding1 = model.encode(cz_to_english_joined, convert_to_tensor=False)
    embedding2 = model.encode(english_joined, convert_to_tensor=False)

    cosine_similarity = util.cos_sim(embedding1, embedding2)

    return(cosine_similarity[0][0].item())


def get_semantic_similarity(czech_sections : list[list[str]], english_sections : list[list[str]]) -> float:

    similarity_sum = 0

    if len(czech_sections) != len(english_sections):
        raise ValueError("The lyrics differ in number of sections.")
    
    n_sections = len(czech_sections)

    n_lines = 0
    for section in czech_sections:
        n_lines += len(section)

    for i in range(n_sections):
        similarity_sum += ((len(czech_sections[i]) / n_lines) * cosine_section_similarity(czech_sections[i], english_sections[i]))
    
    return similarity_sum