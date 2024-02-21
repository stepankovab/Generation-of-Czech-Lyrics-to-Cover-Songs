import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from enum import Enum
import numpy as np
from torch.utils.data import Dataset
from torch.utils.data import Dataset, DataLoader
import os
import json
from transformers import get_linear_schedule_with_warmup
from transformers import AdamW

import logging
import warnings

class DatasetType(Enum):
    BASELINE = 1
    SYLLABLES = 2
    END_OF_LINES = 3
    CHARACTERISTIC_WORD = 4


BATCH_SIZE = 64
MODEL_NAME = "tinyLlama" # "GPT2_oscar" # "GPT2_czech_XL" # "Mistral_czech" # 
EPOCHS = 6
LEARNING_RATE = 3e-5
WARMUP_STEPS = 500
DATASET_PATH = './' # "DATA/Velky_zpevnik" # 
DATASET_TYPE = DatasetType.END_OF_LINES


logging.getLogger().setLevel(logging.CRITICAL)
warnings.filterwarnings('ignore')

device = 'cpu'
if torch.cuda.is_available():
    print("cuda available.")
    device = 'cuda'
    
    torch.cuda.empty_cache()

if MODEL_NAME == "GPT2_oscar":
    tokenizer = AutoTokenizer.from_pretrained("lchaloupsky/czech-gpt2-oscar")
    model = AutoModelForCausalLM.from_pretrained("lchaloupsky/czech-gpt2-oscar")
    tokenizer.model_max_length=1024

elif MODEL_NAME == "GPT2_czech_XL":
    tokenizer = AutoTokenizer.from_pretrained("BUT-FIT/Czech-GPT-2-XL-133k")
    model = AutoModelForCausalLM.from_pretrained("BUT-FIT/Czech-GPT-2-XL-133k")
    tokenizer.model_max_length=1024

elif MODEL_NAME == "Mistral_czech":
    tokenizer = AutoTokenizer.from_pretrained("simecek/cswikimistral_0.1")
    model = AutoModelForCausalLM.from_pretrained("simecek/cswikimistral_0.1")
    tokenizer.model_max_length=1024

elif MODEL_NAME == "tinyLlama":
    tokenizer = AutoTokenizer.from_pretrained("TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T")
    model = AutoModelForCausalLM.from_pretrained("TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T")
    tokenizer.model_max_length=1024
    

model = model.to(device)


class LyricsDataset(Dataset):
    def __init__(self, lyrics_dataset_path = DATASET_PATH, dataset_type = DATASET_TYPE):
        super().__init__()

        lyrics_path = os.path.join(lyrics_dataset_path, 'VZ.json')

        self.lyrics_list = []
        
        with open(lyrics_path, "r", encoding="utf-8") as json_file:
            dataset_dict = json.load(json_file)

        if dataset_type == DatasetType.BASELINE:
            for i in dataset_dict:
                self.lyrics_list.append(', '.join(dataset_dict[i]['lyrics']))

        elif dataset_type == DatasetType.SYLLABLES:
            for dat_i in dataset_dict:
                temp = [" ".join([str(x) for x in dataset_dict[dat_i]['syllables']])]
                for lin_i in range(len(dataset_dict[dat_i]['lyrics'])):
                    temp.append(f"{dataset_dict[dat_i]['syllables'][lin_i]} # {dataset_dict[dat_i]['lyrics'][lin_i]}")
                self.lyrics_list.append("\n".join(temp))

                if dat_i == "12":
                    print(self.lyrics_list[-1])
                    
        elif dataset_type == DatasetType.END_OF_LINES:
            for dat_i in dataset_dict:
                temp = [" ".join(dataset_dict[dat_i]['line_endings'])]
                for lin_i in range(len(dataset_dict[dat_i]['lyrics'])):
                    temp.append(f"{dataset_dict[dat_i]['line_endings'][lin_i]} # {dataset_dict[dat_i]['lyrics'][lin_i]}")
                self.lyrics_list.append("\n".join(temp))

                if dat_i == "12":
                    print(self.lyrics_list[-1])
                
        elif dataset_type == DatasetType.CHARACTERISTIC_WORD:
            for i in dataset_dict:
                self.lyrics_list.append(', '.join(dataset_dict[i]['lyrics']))
        else:
            raise Exception(f"We don't support a Dataset type {dataset_type}")
        
    def __len__(self):
        return len(self.lyrics_list)

    def __getitem__(self, item):
        return self.lyrics_list[item]

dataset = LyricsDataset()
lyrics_loader = DataLoader(dataset, batch_size=1, shuffle=True)

model.load_state_dict(state_dict=torch.load(f"./trained_models/{MODEL_NAME}_{DATASET_TYPE.name}_lyricist_1.pt", map_location=torch.device(device)))

model.train()
optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps = WARMUP_STEPS, num_training_steps = -1)
proc_seq_count = 0
sum_loss = 0.0
batch_count = 0

# lora_r = 16 #@param {type:"number"}
# lora_alpha = 32 #@param {type:"number"}
# lora_dropout = 0.05 #@param {type:"number"}

tmp_lyrics_tens = None
models_folder = "trained_models"
if not os.path.exists(models_folder):
    os.mkdir(models_folder)

for epoch in range(2, EPOCHS):
    
    print(f"EPOCH {epoch} started" + '=' * 30)
    
    for idx,lyric in enumerate(lyrics_loader):

        lyric_tens = torch.tensor(tokenizer.encode(lyric[0])).unsqueeze(0).to(device)
            
        outputs = model(lyric_tens, labels=lyric_tens)
        loss, logits = outputs[:2]                        
        loss.backward()
        sum_loss = sum_loss + loss.detach().data
                       
        proc_seq_count = proc_seq_count + 1
        if proc_seq_count == BATCH_SIZE:
            proc_seq_count = 0    
            batch_count += 1
            optimizer.step()
            scheduler.step() 
            optimizer.zero_grad()
            model.zero_grad()

        if batch_count == 100:
            print(f"sum loss {sum_loss}")
            batch_count = 0
            sum_loss = 0.0
            torch.save(model.state_dict(), os.path.join(models_folder, f"{MODEL_NAME}_{DATASET_TYPE.name}_lyricist_{epoch}.pt"))
    
    # Store the model after each epoch to compare the performance of them
    torch.save(model.state_dict(), os.path.join(models_folder, f"{MODEL_NAME}_{DATASET_TYPE.name}_lyricist_{epoch}.pt"))

