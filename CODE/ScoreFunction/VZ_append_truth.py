import requests
import json
from sentence_transformers import SentenceTransformer
import Imports.tagger as tagger
from syllabator import syllabify

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

with open("DATA/Velky_zpevnik/VZ_pure.json", "r", encoding="utf-8") as json_file:
    dataset_dict = json.load(json_file)

rt = tagger.RhymeTagger()
rt.load_model("cs", verbose=False)

model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')  # multi-language model

for dat_i in [str(int_i) for int_i in range(1, 50)]:
    print(dat_i)
    lyrics_section = dataset_dict[dat_i]["lyrics"]

    # lines count
    lines_count = len(lyrics_section)
    dataset_dict[dat_i]["len"] = lines_count

    # rhyme scheme
    rhymes = rt.tag(poem=lyrics_section, output_format=3)
    rhymes = fill_in_none_rhymes(rhymes)
    dataset_dict[dat_i]["rhymes"] = rhymes

    # syllables count
    sylls_per_line = [len(syllabify(lyrics_section[sec_i], "cs")) for sec_i in range(lines_count)]
    dataset_dict[dat_i]["syllables"] = sylls_per_line
    
    # sentence transformer embedding
    lyrics_joined = ", ".join(lyrics_section)    
    url = 'http://lindat.mff.cuni.cz/services/translation/api/v2/models/cs-en'
    response = requests.post(url, data = {"input_text": lyrics_joined})
    response.encoding='utf8'
    en_lyrics_joined = response.text
    embedding = model.encode(en_lyrics_joined, convert_to_numpy=True)
    dataset_dict[dat_i]["transf_embedding"] = embedding.tolist()


with open("DATA\\Velky_zpevnik\\VZ.json", "w", encoding='utf-8') as json_file:
    json.dump(dataset_dict, json_file, ensure_ascii=False)


# 5, 6 A , 4 B , 6 C , 4 D , 6 E , vector
# line_count, syllables and rhymes, semantics