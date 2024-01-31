from nltk.tokenize import word_tokenize
import lzma
import pickle

# word_count_dict = {}

# with open("DATA\\Velky_zpevnik\\velkyzpevnik.txt", "r", encoding="utf-8") as file:
#     lines = file.readlines()

# for line in lines:
#     line_words = word_tokenize(line, "czech")
#     for word in line_words:
#         if word in word_count_dict:
#             word_count_dict[word] += 1
#         else:
#             word_count_dict[word] = 1

# sorted_word_count = sorted(word_count_dict.items(), key=lambda x:x[1], reverse=True)
# sorted_word_count_dict = dict(sorted_word_count)

# # Serialize the model.
# with lzma.open("DATA/Velky_zpevnik/OG_file_analysis.dict", "wb") as model_file:
#     pickle.dump(sorted_word_count_dict, model_file)


with lzma.open("DATA/Velky_zpevnik/OG_file_analysis.dict", "rb") as model_file:
    sorted_word_count_dict : dict[str, int] = pickle.load(model_file)

c = 0

for word, count in sorted_word_count_dict.items():
    print(word, count)
    c += 1

    if count < 240 and c % 20 == 0:
        pass