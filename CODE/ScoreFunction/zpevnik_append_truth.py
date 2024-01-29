import os
import re
import requests
from sentence_transformers import SentenceTransformer
import Imports.tagger as tagger
from syllabator import syllabify


directory = 'DATA/Velky_zpevnik/VZ_sections'

rt = tagger.RhymeTagger()
rt.load_model("cs", verbose=False)

model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')  # multi-language model
 
# iterate over files in
# that directory
for filename in os.listdir(directory):
    file_path = os.path.join(directory, filename)
    
    with open(file_path, "r", encoding="utf-8") as f:
        lyrics_section = f.readlines()

    lines_count = len(lyrics_section)
    lines_info = []

    rhymes = rt.tag(poem=lyrics_section, output_format=3)

    for i in range(len(lyrics_section)):
        lines_info.append((len(syllabify(lyrics_section[i], "cs")), rhymes[i]))
    
    lyrics_joined = ", ".join(lyrics_section)    

    url = 'http://lindat.mff.cuni.cz/services/translation/api/v2/models/cs-en'
    response = requests.post(url, data = {"input_text": lyrics_joined})
    response.encoding='utf8'
    en_lyrics_joined = response.text

    embedding = model.encode(en_lyrics_joined, convert_to_numpy=True)

    with open("DATA/Velky_zpevnik/VZ_prompted_sections/GT_" + filename, "w", encoding="utf-8") as new_section:
        new_section.write(str(lines_count) + " , " + " , ".join([str(l) + " " + str(r) for (l, r) in lines_info]) + " , " + re.sub("\n", "", str(embedding)))
        new_section.write("\n\n-----\n\n")
        new_section.writelines(lyrics_section)
        



# 5, 6 A , 4 B , 6 C , 4 D , 6 E , vector
# line_count, syllables and rhymes, semantics