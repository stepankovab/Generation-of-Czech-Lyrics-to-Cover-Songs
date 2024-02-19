import requests
import json
import os
from sentence_transformers import SentenceTransformer
import Imports.tagger as tagger
from syllabator import syllabify
import re

HT_PATH = "./DATA"

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

with open(os.path.join(HT_PATH, 'HT.json'), "r", encoding="utf-8") as json_file:
    HT_dataset = json.load(json_file)

# rt_cs = tagger.RhymeTagger()
# rt_cs.load_model("cs", verbose=False)
# rt_en = tagger.RhymeTagger()
# rt_en.load_model("en", verbose=False)

# model = SentenceTransformer('all-MiniLM-L12-v2') 
# HT_dataset["cs"]["num_sections"] = 0
for musical_name in HT_dataset["cs"]:  
    if musical_name == "num_sections":
            continue
      
    print(musical_name)
    # HT_dataset["cs"][musical_name]["num_sections"] = 0
    for song_name in HT_dataset["cs"][musical_name]:
        if song_name == "num_sections":
            continue

        print(song_name)
        
        # HT_dataset["cs"][musical_name]["num_sections"] += len(HT_dataset["cs"][musical_name][song_name])
        # HT_dataset["cs"]["num_sections"] += len(HT_dataset["cs"][musical_name][song_name])  

        for sec_i in HT_dataset["cs"][musical_name][song_name]:
            lyrics_section = HT_dataset["cs"][musical_name][song_name][sec_i]["lyrics"]
            
            # # lines count
            lines_count = len(lyrics_section)
            # HT_dataset["cs"][musical_name][song_name][sec_i]["len"] = lines_count

            # # rhyme scheme
            # rhymes = rt_cs.tag(poem=lyrics_section, output_format=3)
            # rhymes = fill_in_none_rhymes(rhymes)
            # HT_dataset["cs"][musical_name][song_name][sec_i]["rhymes"] = rhymes

            # line endings
            sylls_on_line = [syllabify(lyrics_section[sec_i], "cs") for sec_i in range(lines_count)]
            HT_dataset["cs"][musical_name][song_name][sec_i]["line_endings"] = [syllabified_line[-1][-min(3, len(syllabified_line[-1])):] for syllabified_line in sylls_on_line if len(syllabified_line) > 0]

            # # syllables count
            # HT_dataset["cs"][musical_name][song_name][sec_i]["syllables"] = [len(syllabified_line) for syllabified_line in sylls_on_line]
            
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
            # HT_dataset[dat_i]["transf_embedding"] = embedding.tolist()


with open(os.path.join(HT_PATH, 'HT_added.json'), "w", encoding='utf-8') as json_file:
    json.dump(HT_dataset, json_file, ensure_ascii=False)


# 5, 6 A , 4 B , 6 C , 4 D , 6 E , vector
# line_count, syllables and rhymes, semantics