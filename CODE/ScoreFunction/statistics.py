import lzma
import pickle
import scipy.stats as stats

with lzma.open("C:/Users/barca/MOJE/UMEA/NLP-Umea/FinalProject/DATA/results.dict", "rb") as model_file:
    scores = pickle.load(model_file)

parameters = len(scores["HT"][0])
songs = len(scores["HT"])
averages_HT = []
averages_cs_to_en = []
averages_en_to_cs = []

i = 0

for j in range(songs):
    if i == 1:
        if j == 18:
            continue
        if j == 27:
            continue
    averages_HT.append(max(scores["HT"][j][i], 0))
    averages_cs_to_en.append(max(scores["cz_to_en"][j][i],0))
    averages_en_to_cs.append(max(scores["en_to_cz"][j][i],0))

alpha = 0.05

p_value = stats.ttest_rel(
    averages_HT,
    averages_cs_to_en,
    alternative="greater"
    ).pvalue

print("p value is " + str(p_value))

if p_value <= alpha:
    print('Reject H0 - Alternative holds true')
else:
    print('H0 holds true')

p_value = stats.ttest_rel(
    averages_HT,
    averages_en_to_cs,
    alternative="greater"
    ).pvalue

print("p value is " + str(p_value))

if p_value <= alpha:
    print('Reject H0 - Alternative holds true')
else:
    print('H0 holds true')

p_value = stats.ttest_rel(
    averages_cs_to_en,
    averages_en_to_cs,
    alternative="two-sided"
    ).pvalue

print("p value is " + str(p_value))

if p_value <= alpha:
    print('Reject H0 - Alternative holds true')
else:
    print('H0 holds true')


print("HT", sum(averages_HT)/songs)
print("cz_to_en", sum(averages_cs_to_en)/songs)
print("en_to_cz", sum(averages_en_to_cs)/songs)
print("\n")

averages_HT.clear()
averages_cs_to_en.clear()
averages_en_to_cs.clear()
