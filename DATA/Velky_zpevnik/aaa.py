import json
import os


words = 0
for name in os.listdir("CODE\\NgramStructureFiller\\LyricsGeneratorApp\\Texts"):
    with open(os.path.join("CODE\\NgramStructureFiller\\LyricsGeneratorApp\\Texts", name), "rb") as file:
        dict = file.read()

    words += len(dict.split())

    print(words)


# file_text = ""

# for a in dict:
#     file_text = file_text + " ".join(dict[a]["lyrics"]) + "\n"

# with open("DATA/Velky_zpevnik/VZ_just_lyrics.txt", "w", encoding="utf-8") as file:
#     file.write(file_text)