print("Hello world")
a = 6
print(a + 1)

with open("krasny_vysledek.txt", "a", encoding="utf-8") as f:
    f.writelines(["\n\n ale ted uz vazne\n", "pls\n"])
    