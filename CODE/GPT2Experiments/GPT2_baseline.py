import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel
import numpy as np
from torch.utils.data import Dataset
from torch.utils.data import Dataset, DataLoader
import os
import json
from transformers import get_linear_schedule_with_warmup
from transformers import AdamW

import logging
import warnings


BATCH_SIZE = 16
EPOCHS = 5
LEARNING_RATE = 3e-5
WARMUP_STEPS = 5000
DATASET_PATH = '../../../storage-brno2/home/stepanb2'


logging.getLogger().setLevel(logging.CRITICAL)
warnings.filterwarnings('ignore')

device = 'cpu'
if torch.cuda.is_available():
    print("cuda available.")
    device = 'cuda'

tokenizer = GPT2Tokenizer.from_pretrained("lchaloupsky/czech-gpt2-oscar")
model = GPT2LMHeadModel.from_pretrained("lchaloupsky/czech-gpt2-oscar")
model = model.to(device)
tokenizer.model_max_length=1024


def choose_from_top(probs, n=5):
    ind = np.argpartition(probs, -n)[-n:]
    top_prob = probs[ind]
    top_prob = top_prob / np.sum(top_prob) # Normalize
    choice = np.random.choice(n, 1, p = top_prob)
    token_id = ind[choice][0]
    return int(token_id)



class LyricsDataset(Dataset):
    def __init__(self, lyrics_dataset_path = DATASET_PATH):
        super().__init__()

        lyrics_path = os.path.join(lyrics_dataset_path, 'VZ.json')

        self.lyrics_list = []
        self.end_of_text_token = "<|endoftext|>"
        
        with open(lyrics_path, "r", encoding="utf-8") as json_file:
            dataset_dict = json.load(json_file)

        for i in dataset_dict:
            lyric_str = f"{', '.join(dataset_dict[i]['lyrics'])}{self.end_of_text_token}"
            self.lyrics_list.append(lyric_str)
        
    def __len__(self):
        return len(self.lyrics_list)

    def __getitem__(self, item):
        return self.lyrics_list[item]

dataset = LyricsDataset()
joke_loader = DataLoader(dataset, batch_size=1, shuffle=True)



model.train()
optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps = WARMUP_STEPS, num_training_steps = -1)
proc_seq_count = 0
sum_loss = 0.0
batch_count = 0

tmp_jokes_tens = None
models_folder = "trained_models"
if not os.path.exists(models_folder):
    os.mkdir(models_folder)

for epoch in range(EPOCHS):
    
    print(f"EPOCH {epoch} started" + '=' * 30)
    
    for idx,joke in enumerate(joke_loader):

        joke_tens = torch.tensor(tokenizer.encode(joke[0])).unsqueeze(0).to(device)
            
        outputs = model(joke_tens, labels=joke_tens)
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
            torch.save(model.state_dict(), os.path.join(models_folder, f"gpt2_medium_joker_{epoch}_{sum_loss}.pt"))
    
    # Store the model after each epoch to compare the performance of them
    torch.save(model.state_dict(), os.path.join(models_folder, f"gpt2_medium_joker_{epoch}.pt"))

