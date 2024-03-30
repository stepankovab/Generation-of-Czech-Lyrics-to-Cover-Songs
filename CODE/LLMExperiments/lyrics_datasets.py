import os
import json
import re
from dataset_types import DatasetType
from torch.utils.data import Dataset
from english_structure_extractor import SectionStructure
from eval.syllabator import dashed_syllabified_line, syllabify

class LinesLyricsDataset(Dataset):
    def __init__(self, lyrics_dataset_path, dataset_type):
        super().__init__()

        lyrics_path = os.path.join(lyrics_dataset_path, 'VZ.json')

        self.lines_list = []
        
        with open(lyrics_path, "r", encoding="utf-8") as json_file:
            dataset_dict = json.load(json_file)

        if dataset_type == DatasetType.BASELINE:
            for dat_i in dataset_dict:
                for line in dataset_dict[dat_i]['lyrics']:
                    self.lines_list.append(line + "\n")

        elif dataset_type == DatasetType.SYLLABLES:
            for dat_i in dataset_dict:
                for lin_i in range(len(dataset_dict[dat_i]['lyrics'])):
                    self.lines_list.append(f"{dataset_dict[dat_i]['syllables'][lin_i]} # {dataset_dict[dat_i]['lyrics'][lin_i]}\n")
        
        elif dataset_type == DatasetType.SYLLABLES_ENDS:
            for dat_i in dataset_dict:
                for lin_i in range(len(dataset_dict[dat_i]['lyrics'])):
                    self.lines_list.append(f"{dataset_dict[dat_i]['syllables'][lin_i]} # {dataset_dict[dat_i]['line_endings'][lin_i]} # {dataset_dict[dat_i]['lyrics'][lin_i]}\n")
                   
        elif dataset_type == DatasetType.WORDS:
            for dat_i in dataset_dict:
                for lin_i in range(len(dataset_dict[dat_i]['lyrics'])):
                    if dataset_dict[dat_i]['line_keywords'][lin_i] == '':
                        continue
                    self.lines_list.append(f"{dataset_dict[dat_i]['line_keywords'][lin_i]} # {dataset_dict[dat_i]['lyrics'][lin_i]}\n")

        elif dataset_type == DatasetType.WORDS_ENDS:
            for dat_i in dataset_dict:
                for lin_i in range(len(dataset_dict[dat_i]['lyrics'])):
                    if dataset_dict[dat_i]['line_keywords'][lin_i] == '':
                        continue
                    self.lines_list.append(f"{dataset_dict[dat_i]['line_keywords'][lin_i]}\n{dataset_dict[dat_i]['line_endings'][lin_i]} # {dataset_dict[dat_i]['lyrics'][lin_i]}\n")

        elif dataset_type == DatasetType.SYLLABLES_WORDS:
            for dat_i in dataset_dict:
                for lin_i in range(len(dataset_dict[dat_i]['lyrics'])):
                    if dataset_dict[dat_i]['line_keywords'][lin_i] == '':
                        continue
                    self.lines_list.append(f"{dataset_dict[dat_i]['line_keywords'][lin_i]}\n{dataset_dict[dat_i]['syllables'][lin_i]} # {dataset_dict[dat_i]['lyrics'][lin_i]}\n")

        elif dataset_type == DatasetType.SYLLABLES_WORDS_ENDS:
            for dat_i in dataset_dict:
                for lin_i in range(len(dataset_dict[dat_i]['lyrics'])):
                    if dataset_dict[dat_i]['line_keywords'][lin_i] == '':
                        continue
                    self.lines_list.append(f"{dataset_dict[dat_i]['line_keywords'][lin_i]}\n{dataset_dict[dat_i]['syllables'][lin_i]} # {dataset_dict[dat_i]['line_endings'][lin_i]} # {dataset_dict[dat_i]['lyrics'][lin_i]}\n")

        elif dataset_type == DatasetType.UNRHYMED_LEN:
            for dat_i in dataset_dict:
                if len(dataset_dict[dat_i]["lyrics"]) != len(dataset_dict[dat_i]["unrhymed"]):
                    continue
                for lin_i in range(len(dataset_dict[dat_i]['lyrics'])):
                    unrhymed = dataset_dict[dat_i]["unrhymed"][lin_i]
                    original = dataset_dict[dat_i]["lyrics"][lin_i]

                    unrhymed_len = len(syllabify(unrhymed))

                    if unrhymed_len == dataset_dict[dat_i]["syllables"][lin_i]:
                        original = unrhymed

                    self.lines_list.append(f"{unrhymed_len} # {unrhymed}\n{dataset_dict[dat_i]['syllables'][lin_i]} # {original}\n")

        elif dataset_type == DatasetType.UNRHYMED_LEN_END:
            for dat_i in dataset_dict:
                if len(dataset_dict[dat_i]["lyrics"]) != len(dataset_dict[dat_i]["unrhymed"]):
                    continue
                for lin_i in range(len(dataset_dict[dat_i]['lyrics'])):
                    unrhymed = dataset_dict[dat_i]["unrhymed"][lin_i]
                    original = dataset_dict[dat_i]["lyrics"][lin_i]

                    unrhymed_sylls = syllabify(unrhymed)
                    unrhymed_len = len(unrhymed_sylls)
                    if unrhymed_len == 0:
                        continue

                    unrhymed_end = unrhymed_sylls[-1][-min(len(unrhymed_sylls[-1]), 3):]

                    if unrhymed_end == dataset_dict[dat_i]["line_endings"][lin_i] and unrhymed_len == dataset_dict[dat_i]["syllables"][lin_i]:
                        original = unrhymed

                    self.lines_list.append(f"{unrhymed_len} # {unrhymed_end} # {unrhymed}\n{dataset_dict[dat_i]['syllables'][lin_i]} # {dataset_dict[dat_i]['line_endings'][lin_i]} # {original}\n")

        elif dataset_type == DatasetType.FORCED_SYLLABLES:
            for dat_i in dataset_dict:
                for lin_i in range(len(dataset_dict[dat_i]['lyrics'])):
                    self.lines_list.append(f"{dataset_dict[dat_i]['syllables'][lin_i]} # {dashed_syllabified_line(dataset_dict[dat_i]['lyrics'][lin_i])}\n")

        elif dataset_type == DatasetType.FORCED_SYLLABLES_ENDS:
            for dat_i in dataset_dict:
                for lin_i in range(len(dataset_dict[dat_i]['lyrics'])):
                    self.lines_list.append(f"{dataset_dict[dat_i]['syllables'][lin_i]} # {dataset_dict[dat_i]['line_endings'][lin_i]} # {dashed_syllabified_line(dataset_dict[dat_i]['lyrics'][lin_i])}\n")

        else:
            raise Exception(f"We don't support a Dataset type {dataset_type}")
        
    def __len__(self):
        return len(self.lines_list)

    def __getitem__(self, item):
        return self.lines_list[item]
    


class WholeLyricsDataset(Dataset):
    def __init__(self, lyrics_dataset_path, dataset_type):
        super().__init__()

        lyrics_path = os.path.join(lyrics_dataset_path, 'VZ.json')

        self.lyrics_list = []
        
        with open(lyrics_path, "r", encoding="utf-8") as json_file:
            dataset_dict = json.load(json_file)

        if dataset_type == DatasetType.BASELINE:
            for i in dataset_dict:
                self.lyrics_list.append('\n'.join(dataset_dict[i]['lyrics']))

        elif dataset_type == DatasetType.SYLLABLES:
            for dat_i in dataset_dict:
                temp = [f"{' '.join([str(x) for x in dataset_dict[dat_i]['syllables']])} #"]
                for lin_i in range(len(dataset_dict[dat_i]['lyrics'])):
                    temp.append(f"{dataset_dict[dat_i]['syllables'][lin_i]} # {dataset_dict[dat_i]['lyrics'][lin_i]}")
                self.lyrics_list.append("\n".join(temp) + "\n")
                    
        elif dataset_type == DatasetType.WORDS:
            for dat_i in dataset_dict:
                temp = [f"{dataset_dict[dat_i]['len']} # {' '.join(dataset_dict[dat_i]['keywords'])} #"]
                for lin_i in range(len(dataset_dict[dat_i]['lyrics'])):
                    temp.append(f"{dataset_dict[dat_i]['lyrics'][lin_i]}")
                self.lyrics_list.append("\n".join(temp) + "\n")

        elif dataset_type == DatasetType.SYLLABLES_WORDS:
            for dat_i in dataset_dict:
                temp = [f"{' '.join([str(x) for x in dataset_dict[dat_i]['syllables']])} # {' '.join(dataset_dict[dat_i]['keywords'])} #"]
                for lin_i in range(len(dataset_dict[dat_i]['lyrics'])):
                    temp.append(f"{dataset_dict[dat_i]['syllables'][lin_i]} # {dataset_dict[dat_i]['lyrics'][lin_i]}")
                self.lyrics_list.append("\n".join(temp) + "\n")

        elif dataset_type == DatasetType.FORCED_SYLLABLES:
            for dat_i in dataset_dict:
                temp = [f"{' '.join([str(x) for x in dataset_dict[dat_i]['syllables']])} #"]
                for lin_i in range(len(dataset_dict[dat_i]['lyrics'])):
                    temp.append(f"{dataset_dict[dat_i]['syllables'][lin_i]} # {dashed_syllabified_line(dataset_dict[dat_i]['lyrics'][lin_i])}")
                self.lyrics_list.append("\n".join(temp) + "\n")

        elif dataset_type == DatasetType.RHYME_SCHEME:
            for dat_i in dataset_dict:
                temp = [f"{' '.join([str(x) for x in dataset_dict[dat_i]['rhymes']])} #"]
                for lin_i in range(len(dataset_dict[dat_i]['lyrics'])):
                    temp.append(f"{dataset_dict[dat_i]['rhymes'][lin_i]} # {dataset_dict[dat_i]['lyrics'][lin_i]}")
                self.lyrics_list.append("\n".join(temp) + "\n")

        elif dataset_type == DatasetType.SYLLABLES_RHYME_SCHEME:
            for dat_i in dataset_dict:
                temp = [f"{' '.join([str(x) for x in dataset_dict[dat_i]['syllables']])} # {' '.join([str(x) for x in dataset_dict[dat_i]['rhymes']])} #"]
                for lin_i in range(len(dataset_dict[dat_i]['lyrics'])):
                    temp.append(f"{dataset_dict[dat_i]['syllables'][lin_i]} # {dataset_dict[dat_i]['rhymes'][lin_i]} # {dataset_dict[dat_i]['lyrics'][lin_i]}")
                self.lyrics_list.append("\n".join(temp) + "\n")

        elif dataset_type == DatasetType.SYLLABLES_RHYME_SCHEME_WORDS:
            for dat_i in dataset_dict:
                temp = [f"{' '.join([str(x) for x in dataset_dict[dat_i]['syllables']])} # {' '.join([str(x) for x in dataset_dict[dat_i]['rhymes']])} # {' '.join(dataset_dict[dat_i]['keywords'])} #"]
                for lin_i in range(len(dataset_dict[dat_i]['lyrics'])):
                    temp.append(f"{dataset_dict[dat_i]['syllables'][lin_i]} # {dataset_dict[dat_i]['rhymes'][lin_i]} # {dataset_dict[dat_i]['lyrics'][lin_i]}")
                self.lyrics_list.append("\n".join(temp) + "\n")

        else:
            raise Exception(f"We don't support a Dataset type {dataset_type}")
        
    def __len__(self):
        return len(self.lyrics_list)

    def __getitem__(self, item):
        return self.lyrics_list[item]
    

def prepare_prompt_whole(dataset_type, structure: SectionStructure):
    if dataset_type == DatasetType.BASELINE:
            prompt = " "
    elif dataset_type == DatasetType.SYLLABLES:
        prompt = f"{' '.join([str(x) for x in structure.syllables])} #\n"
    elif dataset_type == DatasetType.WORDS:
        prompt = f"{structure.num_lines} # {' '.join(structure.keywords)} #\n"
    elif dataset_type == DatasetType.SYLLABLES_WORDS:
        prompt = f"{' '.join([str(x) for x in structure.syllables])} # {' '.join(structure.keywords)} #\n"
    elif dataset_type == DatasetType.FORCED_SYLLABLES:
        prompt = f"{' '.join([str(x) for x in structure.syllables])} #\n"
    elif dataset_type == DatasetType.RHYME_SCHEME:
        prompt = f"{' '.join([str(x) for x in structure.rhyme_scheme])} #\n"
    elif dataset_type == DatasetType.SYLLABLES_RHYME_SCHEME:
        prompt = f"{' '.join([str(x) for x in structure.syllables])} # {' '.join([str(x) for x in structure.rhyme_scheme])} #\n"
    elif dataset_type == DatasetType.SYLLABLES_RHYME_SCHEME_WORDS:
        prompt = f"{' '.join([str(x) for x in structure.syllables])} # {' '.join([str(x) for x in structure.rhyme_scheme])} # {' '.join(structure.keywords)} #\n"
    else:
        raise Exception(f"We don't support a Dataset type {dataset_type}")

    return prompt


def extract_output_whole(out_lines, prompt, dataset_type, structure):
    
    print("+++++++++++++++++++++++++\ndetokenized text: ")
    print(out_lines.replace(prompt,''))
    print()

    out_lines = out_lines.strip().split("\n")
    start_of_text = len(prompt.strip().split("\n"))
    if not prompt.strip():
        start_of_text = 0

    output = []
    for line_i in range(start_of_text, min(len(out_lines), structure.num_lines + start_of_text)):
        line = out_lines[line_i]
        line = re.sub(',', '', line)
        if not line.strip():
            output.append("")
            continue
    
        line_sections = line.strip().split("#")
        if dataset_type in [DatasetType.SYLLABLES, DatasetType.SYLLABLES_WORDS, DatasetType.FORCED_SYLLABLES, DatasetType.RHYME_SCHEME]:
            if len(line_sections) > 1:
                line = ' # '.join([x.strip() for x in line_sections[1:]])
        if dataset_type in [DatasetType.SYLLABLES_RHYME_SCHEME, DatasetType.SYLLABLES_RHYME_SCHEME_WORDS]:
            if len(line_sections) > 2:
                line = ' # '.join([x.strip() for x in line_sections[2:]])
            elif len(line_sections) > 1:
                line = ' # '.join([x.strip() for x in line_sections[1:]])

        output.append(line)

    return output