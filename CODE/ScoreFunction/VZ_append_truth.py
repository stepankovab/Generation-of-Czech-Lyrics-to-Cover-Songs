import requests
import json
from sentence_transformers import SentenceTransformer
import Imports.tagger as tagger
from syllabator import syllabify
import re

def fill_in_none_rhymes(rhymes : list[int|None]) -> list[str]:
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

with open("DATA/Velky_zpevnik/VZ.json", "r", encoding="utf-8") as json_file:
    dataset_dict = json.load(json_file)

# rt = tagger.RhymeTagger()
# rt.load_model("cs", verbose=False)

# model = SentenceTransformer('all-MiniLM-L12-v2') 

for dat_i in dataset_dict:   
    lyrics_section = dataset_dict[dat_i]["lyrics"]
    without_newlines_section = []

    # # remove new lines at the and of the lines
    # for line in lyrics_section:
    #     while len(line) > 0 and re.match(r'[\s,\.()]+', line[-1]):
    #         line = line[:-1]
    #     without_newlines_section.append(line)
    
    # dataset_dict[dat_i]["lyrics"] = without_newlines_section

    # # lines count
    lines_count = len(lyrics_section)
    # dataset_dict[dat_i]["len"] = lines_count

    # # rhyme scheme
    # rhymes = rt.tag(poem=lyrics_section, output_format=3)
    # rhymes = fill_in_none_rhymes(rhymes)
    # dataset_dict[dat_i]["rhymes"] = rhymes

    # line endings
    sylls_on_line = [syllabify(lyrics_section[sec_i], "cs")for sec_i in range(lines_count)]
    dataset_dict[dat_i]["line_endings"] = [syllabified_line[-1][-min(3, len(syllabified_line[-1])):] for syllabified_line in sylls_on_line if len(syllabified_line) > 0]

    # # syllables count
    # dataset_dict[dat_i]["syllables"] = [len(syllabified_line) for syllabified_line in sylls_on_line]
    
    # # sentence transformer embedding
    # lyrics_joined = ", ".join(lyrics_section)    
    # url = 'http://lindat.mff.cuni.cz/services/translation/api/v2/models/cs-en'
    # response = requests.post(url, data = {"input_text": lyrics_joined})
    # response.encoding='utf8'
    # en_lyrics_joined = response.text

    # url = 'http://lindat.mff.cuni.cz/services/ker'
    # response = requests.post(url, data="file=@pokus.txt")
    # response.encoding='utf8'
    # keywords = response.text


    # embedding = model.encode(en_lyrics_joined, convert_to_numpy=True)
    # dataset_dict[dat_i]["transf_embedding"] = embedding.tolist()


with open("DATA\\Velky_zpevnik\\VZ_added.json", "w", encoding='utf-8') as json_file:
    json.dump(dataset_dict, json_file, ensure_ascii=False)


# 5, 6 A , 4 B , 6 C , 4 D , 6 E , vector
# line_count, syllables and rhymes, semantics