from evaluator import Evaluator
from english_structure_extractor import SectionStructure
from eval.syllabator import syllabify
import re
import random


class Postprocesser():
    def __init__(self, evaluator = Evaluator()) -> None:
        self.evaluator = evaluator
        self.stopwords = {"a","ačkoli","ahoj","ale","anebo","ano","asi","aspoň","během","bez","beze","blízko","bohužel","brzo","bude","budeme","budeš","budete","budou","budu","byl","byla","byli","bylo","byly","bys","čau","chce","chceme","chceš","chcete","chci","chtějí","chtít","chut'","chuti","co","čtrnáct","čtyři","dál","dále","daleko","děkovat","děkujeme","děkuji","den","deset","devatenáct","devět","do","dobrý","docela","dva","dvacet","dvanáct","dvě","hodně","já","jak","jde","je","jeden","jedenáct","jedna","jedno","jednou","jedou","jeho","její","jejich","jemu","jen","jenom","ještě","jestli","jestliže","jí","jich","jím","jimi","jinak","jsem","jsi","jsme","jsou","jste","kam","kde","kdo","kdy","když","kolik","kromě","která","které","kteří","který","kvůli","má","mají","málo","mám","máme","máš","máte","mé","mě","mezi","mí","mít","mně","mffjeez","mnou","moc","mohl","mohou","moje","moji","možná","můj","musí","může","my","na","nad","nade","nám","námi","naproti","nás","náš","naše","naši","ne","ně","nebo","nebyl","nebyla","nebyli","nebyly","něco","nedělá","nedělají","nedělám","neděláme","neděláš","neděláte","nějak","nejsi","někde","někdo","nemají","nemáme","nemáte","neměl","němu","není","nestačí","nevadí","než","nic","nich","ním","nimi","nula","od","ode","on","ona","oni","ono","ony","osm","osmnáct","pak","patnáct","pět","po","pořád","potom","pozdě","před","přes","přese","pro","proč","prosím","prostě","proti","protože","rovně","se","sedm","sedmnáct","šest","šestnáct","skoro","smějí","smí","snad","spolu","sta","sté","sto","ta","tady","tak","takhle","taky","tam","tamhle","tamhleto","tamto","tě","tebe","tebou","ted'","tedy","ten","ti","tisíc","tisíce","to","tobě","tohle","toto","třeba","tři","třináct","trošku","tvá","tvé","tvoje","tvůj","ty","určitě","už","vám","vámi","vás","váš","vaše","vaši","večer","vedle","vlastně","všechno","všichni","vůbec","vy","vždy","za","zač","zatímco","ze","že","aby","aj","ani","az","budem","budes","by","byt","ci","clanek","clanku","clanky","coz","cz","dalsi","design","dnes","email","ho","jako","jej","jeji","jeste","ji","jine","jiz","jses","kdyz","ktera","ktere","kteri","kterou","ktery","ma","mate","mi","mit","muj","muze","nam","napiste","nas","nasi","nejsou","neni","nez","nove","novy","pod","podle","pokud","pouze","prave","pred","pres","pri","proc","proto","protoze","prvni","pta","re","si","strana","sve","svych","svym","svymi","take","takze","tato","tema","tento","teto","tim","timto","tipy","toho","tohoto","tom","tomto","tomuto","tu","tuto","tyto","uz","vam","vas","vase","vice","vsak","zda","zde","zpet","zpravy","a","aniž","až","být","což","či","článek","článku","články","další","i","jenž","jiné","již","jseš","jšte","každý","kteři","ku","me","ná","napište","nechť","ní","nové","nový","o","práve","první","přede","při","sice","své","svůj","svých","svým","svými","také","takže","te","těma","této","tím","tímto","u","více","však","všechen","z","zpět","zprávy"}

    def choose_best_line(self, lines_list, syll_dist_tolerance=0.1, sem_sim_tolerance=0.6, syllables_in = None, ending_in = None, text_in = None, text_in_english=False, remove_add_stopwords=False):
        if len(lines_list) == 0:
            return ""
        
        if remove_add_stopwords and syllables_in != None:
            lines_list = self.correct_length_remove_add_stopwords(lines_list, [syllables_in for _ in range(len(lines_list))], ending_in != None)

        if len(lines_list) == 1:
            return lines_list[0]
        if syllables_in == None and ending_in == None and text_in == None:
            return lines_list[0]
            
        scores = [0 for _ in range(len(lines_list))]

        # fill in scores dict
        for line_i in range(len(lines_list)):
            sylls = syllabify(lines_list[line_i])
            
            if syllables_in != None:
                syllables_out = len(sylls)
                syll_dist = self.evaluator.get_section_syllable_distance([syllables_in], [syllables_out])
                scores[line_i] += syll_dist / max(syll_dist_tolerance, 0.0001)
            
            if ending_in != None:
                if len(sylls) == 0:
                    rhyme_score = 0
                else:
                    ending_out = sylls[-1]
                    scheme = self.evaluator.czech_rhyme_detector.tag([ending_in, ending_out])
                    if scheme[0] == scheme[1] and scheme[0] != None:
                        rhyme_score = 0
                    else:
                        rhyme_score = 1
                scores[line_i] += rhyme_score

            if text_in != None:
                similarity = self.evaluator.get_semantic_similarity(lines_list[line_i], text_in, text1_in_en=False, text2_in_en=text_in_english)
                scores[line_i] += (1 - similarity) / (1 - min(sem_sim_tolerance, 0.9999))

        ordered_indicies = [i for i, x in sorted(enumerate(scores), key=lambda x: x[1])]

        return lines_list[ordered_indicies[0]]

    def choose_best_section(self,
                            lyrics_list,
                            structure : SectionStructure,
                            syll_dist_tolerance=0.1,
                            rhyme_scheme_agree_tolerance=0.4,
                            sem_sim_tolerance=0.6,
                            phon_rep_dif_tolerance=0.15,
                            remove_add_stopwords=False):
        
        if isinstance(lyrics_list, str):
            lyrics_list = [lyrics_list]
            
        if remove_add_stopwords:
            for i in range(len(lyrics_list)):
                lyrics_list[i] = self.correct_length_remove_add_stopwords(lyrics_list[i], structure.syllables)

        if len(lyrics_list) == 0:
            return ""
        if len(lyrics_list) == 1:
            return lyrics_list[0]

        scores_dict = self.evaluator.evaluate_outputs_structure([(','.join(lyrics[:min(len(lyrics), structure.num_lines * 2)]), structure) for lyrics in lyrics_list], evaluate_keywords=False, evaluate_line_keywords=False, evaluate_translations=False)
        scores = [0 for _ in range(len(lyrics_list))]

        # pick the best match
        for line_i in range(len(lyrics_list)):
            scores[line_i] += scores_dict["syll_dist"][line_i] / max(syll_dist_tolerance, 0.0001)
            scores[line_i] += (1 - scores_dict["rhyme_scheme_agree"][line_i]) / (1 - min(rhyme_scheme_agree_tolerance, 0.9999))
            scores[line_i] += (1 - scores_dict["semantic_sim"][line_i]) / (1 - min(sem_sim_tolerance, 0.9999))
            scores[line_i] += scores_dict["phon_rep_dif"][line_i] / max(phon_rep_dif_tolerance, 0.0001)

        ordered_indicies = [i for i, x in sorted(enumerate(scores), key=lambda x: x[1])]

        return lyrics_list[ordered_indicies[0]]

    def correct_length_remove_add_stopwords(self, lines, lengths, keep_last_word=True):
        if isinstance(lines, str):
            lines = [lines]

        if len(lines) != len(lengths):
            return lines
        
        for line_i in range(len(lines)):
            
            line = lines[line_i].lower()
            line_len = lengths[line_i]
            difference = len(syllabify(line)) - line_len

            if difference == 0 or difference > 2 * line_len:
                continue

            if difference > 0:
                words_to_remove = []
                words = line.strip().split(" ")
                for word in words[:-1]:
                    if word in self.stopwords:
                        words_to_remove.append(word)
                if not keep_last_word:
                    if words[-1] in self.stopwords:
                        words_to_remove.append(words[-1])
                line, difference = self._find_best_removal(line, words_to_remove, difference)

            if difference < 0 and line.strip():
                if difference == -1:
                    prefix = random.choice(["a", "tak", "že", "co", "dál"])
                if difference == -2:
                    prefix = random.choice(["že prý", "a tak", "a pak", "copak", "no tak"])
                if difference <= -3:
                    prefix = "na" + " na"*(-difference - 1)
                line = prefix + " " + line

            lines[line_i] = line

        return lines

    def _find_best_removal(self, line, words_to_remove : list, difference, potential_results=None):
        line = re.sub(r'\s+', ' ', line).strip()

        # stopcondition
        if words_to_remove == []:
            if potential_results == None:
                return (line, difference)
            else:
                if abs(difference) < abs(potential_results[1]) or abs(difference) == abs(potential_results[1]) and difference < potential_results[1]:
                    return (line, difference)
                else:
                    return potential_results

        else:   
            word = words_to_remove[0]
            inline_word = word
            if f"v {word}" in line:
                inline_word = f"v {word}"
            elif f"s {word}" in line:
                inline_word = f"s {word}"  
            elif f"k {word}" in line:
                inline_word = f"k {word}" 
            words_to_remove.remove(word)
            potential_results = self._find_best_removal(line, words_to_remove, difference, potential_results)
            potential_results = self._find_best_removal(re.sub(r'\b' + inline_word + r'\b', '', line, 1), words_to_remove, difference - len(syllabify(word)), potential_results)
            words_to_remove.append(word)

            if potential_results == None:
                return (line, difference)
            else:
                if abs(difference) < abs(potential_results[1]) or abs(difference) == abs(potential_results[1]) and difference < potential_results[1]:
                    return (line, difference)
                else:
                    return potential_results
