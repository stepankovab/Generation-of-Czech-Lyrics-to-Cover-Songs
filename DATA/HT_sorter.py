import json


with open("english_HT.txt", "r", encoding="utf8") as efile:
    e = efile.readlines()

with open("czech_HT.txt", "r", encoding="utf8") as cfile:
    c = cfile.readlines()


assert len(e) == len(c)

HT = {"cs" : {}, "en" : {}}

song_i = 0
sec_i = 0
song_cs = {}
song_en = {}
sec_cs = []
sec_en = []

for line_i in range(len(e)):
    if "=" in e[line_i]:
        assert "=" in c[line_i]

        if len(sec_cs) == 0:
            raise ValueError
        if len(sec_en) == 0:
            raise ValueError

        song_cs[sec_i] = sec_cs.copy()
        song_en[sec_i] = sec_en.copy()

        sec_cs.clear()
        sec_en.clear()
        sec_i += 1

        if len(song_cs) == 0:
            raise ValueError
        if len(song_en) == 0:
            raise ValueError

        HT["cs"][song_i] = song_cs.copy()
        HT["en"][song_i] = song_en.copy()

        song_cs.clear()
        song_en.clear()
        song_i += 1

        continue

    if not e[line_i].strip():
        assert not c[line_i].strip()

        if len(sec_cs) == 0:
            raise ValueError
        if len(sec_en) == 0:
            raise ValueError

        song_cs[sec_i] = sec_cs.copy()
        song_en[sec_i] = sec_en.copy()

        sec_cs.clear()
        sec_en.clear()
        sec_i += 1

        continue

    cstripped = c[line_i].strip().replace(',','')
    estripped = e[line_i].strip().replace(',','')

    if not cstripped.strip():
        raise ValueError
    if not estripped.strip():
        raise ValueError

    sec_cs.append(cstripped)
    sec_en.append(estripped)


with open("new_HT.json", "w", encoding='utf-8') as json_file:
    json.dump(HT, json_file, ensure_ascii=False)
