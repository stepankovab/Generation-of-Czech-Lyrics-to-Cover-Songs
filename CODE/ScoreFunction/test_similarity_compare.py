from sentence_transformers import SentenceTransformer, util
import requests

lyrics = "Zdálo by se, že tvý trable nikdo netuší, i když duše závoj černej má, snad naše píseň o slunci ty slzy vysuší, pozor dávej, refrén začíná. "
url = 'http://lindat.mff.cuni.cz/services/translation/api/v2/models/cs-en'
response = requests.post(url, data = {"input_text": lyrics})
response.encoding='utf8'
en_lyrics = response.text

model = SentenceTransformer('all-MiniLM-L12-v2') 
en_lyrics_embedding = model.encode(en_lyrics, convert_to_tensor=True, normalize_embeddings=True)
cs_lyrics_embedding = model.encode(lyrics, convert_to_tensor=True, normalize_embeddings=True)
print(util.dot_score(en_lyrics_embedding, cs_lyrics_embedding)[0][0], "\n")

# keywords = ["trabl", "závoj", "refrén", "slza", "duše", "slunce", "píseň", "černý"]
keywords = ["trabl", "závoj", "refrén", "slza", "duše", "slunce", "píseň", "černý", "temný závoj písně refrén", "trabl závoj refrén slza duše slunce píseň černý", "trabl závoj refrén slza", "duše slunce píseň černý"]

for keyword in keywords:
    response = requests.post(url, data = {"input_text": keyword})
    response.encoding='utf8'
    en_keyword = response.text
    en_keyword_embedding = model.encode(en_keyword, convert_to_tensor=True, normalize_embeddings=True)
    cs_keyword_embedding = model.encode(keyword, convert_to_tensor=True, normalize_embeddings=True)
    en_similarity = util.dot_score(en_lyrics_embedding, en_keyword_embedding)
    cs_similarity = util.dot_score(cs_lyrics_embedding, cs_keyword_embedding)

    print(keyword, "----", cs_similarity[0][0])
    print(en_keyword[:-1], "----", en_similarity[0][0])
    print()


