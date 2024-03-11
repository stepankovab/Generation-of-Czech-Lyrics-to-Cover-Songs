import os
import json

def HT_loader(dataset_path, language = "en"):
    lyrics_path = os.path.join(dataset_path, 'HT.json')

    with open(lyrics_path, "r", encoding="utf-8") as json_file:
        dataset_dict = json.load(json_file)

    if language == "en" or language == "cs":
        HT_lyrics = _extract_lyrics(dataset_dict, language)

    elif language == "both":
        HT_lyrics = _extract_lyrics(dataset_dict, "cs") + _extract_lyrics(dataset_dict, "en")

    else:
        HT_lyrics = []

    return HT_lyrics

def _extract_lyrics(dataset_dict, language):
    list_of_sections = []

    for mov in dataset_dict[language]:
        for song in dataset_dict[language][mov]:
            for section in dataset_dict[language][mov][song]:
                list_of_sections.append(','.join(dataset_dict[language][mov][song][section]["lyrics"]))

    return list_of_sections