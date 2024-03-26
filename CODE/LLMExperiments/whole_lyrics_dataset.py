import os
import json
from dataset_types import DatasetType
from torch.utils.data import Dataset
from eval.syllabator import dashed_syllabified_line

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
                    
        elif dataset_type == DatasetType.END_OF_LINES:
            for dat_i in dataset_dict:
                temp = [f"{' '.join(dataset_dict[dat_i]['line_endings'])} #"]
                for lin_i in range(len(dataset_dict[dat_i]['lyrics'])):
                    temp.append(f"{dataset_dict[dat_i]['line_endings'][lin_i]} # {dataset_dict[dat_i]['lyrics'][lin_i]}")
                self.lyrics_list.append("\n".join(temp) + "\n")
                
        elif dataset_type == DatasetType.CHARACTERISTIC_WORDS:
            for dat_i in dataset_dict:
                temp = [f"{dataset_dict[dat_i]['len']} # {' '.join(dataset_dict[dat_i]['keywords'])} #"]
                for lin_i in range(len(dataset_dict[dat_i]['lyrics'])):
                    temp.append(f"{lin_i + 1}. # {dataset_dict[dat_i]['lyrics'][lin_i]}")
                self.lyrics_list.append("\n".join(temp) + "\n")

        elif dataset_type == DatasetType.SYLLABLES_AND_WORDS:
            for dat_i in dataset_dict:
                temp = [f"{' '.join([str(x) for x in dataset_dict[dat_i]['syllables']])} # {' '.join(dataset_dict[dat_i]['keywords'])} #"]
                for lin_i in range(len(dataset_dict[dat_i]['lyrics'])):
                    temp.append(f"{dataset_dict[dat_i]['syllables'][lin_i]} # {dataset_dict[dat_i]['lyrics'][lin_i]}")
                self.lyrics_list.append("\n".join(temp) + "\n")

        elif dataset_type == DatasetType.SYLLABLES_AND_ENDS:
            for dat_i in dataset_dict:
                temp = [f"{' '.join([str(x) for x in dataset_dict[dat_i]['syllables']])} # {' '.join(dataset_dict[dat_i]['line_endings'])} #"]
                for lin_i in range(len(dataset_dict[dat_i]['lyrics'])):
                    temp.append(f"{dataset_dict[dat_i]['syllables'][lin_i]} # {dataset_dict[dat_i]['line_endings'][lin_i]} # {dataset_dict[dat_i]['lyrics'][lin_i]}")
                self.lyrics_list.append("\n".join(temp) + "\n")

        elif dataset_type == DatasetType.ENDS_AND_WORDS:
            for dat_i in dataset_dict:
                temp = [f"{' '.join(dataset_dict[dat_i]['line_endings'])} # {' '.join(dataset_dict[dat_i]['keywords'])} #"]
                for lin_i in range(len(dataset_dict[dat_i]['lyrics'])):
                    temp.append(f"{dataset_dict[dat_i]['line_endings'][lin_i]} # {dataset_dict[dat_i]['lyrics'][lin_i]}")
                self.lyrics_list.append("\n".join(temp) + "\n")
            
        elif dataset_type == DatasetType.FORCED_SYLLABLES:
            for dat_i in dataset_dict:
                temp = [f"{' '.join([str(x) for x in dataset_dict[dat_i]['syllables']])} #"]
                for lin_i in range(len(dataset_dict[dat_i]['lyrics'])):
                    temp.append(f"{dataset_dict[dat_i]['syllables'][lin_i]} # {dashed_syllabified_line(dataset_dict[dat_i]['lyrics'][lin_i])}")
                self.lyrics_list.append("\n".join(temp) + "\n")

        else:
            raise Exception(f"We don't support a Dataset type {dataset_type}")
        
    def __len__(self):
        return len(self.lyrics_list)

    def __getitem__(self, item):
        return self.lyrics_list[item]