import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from dataset_types import DatasetType
from torch.utils.data import Dataset
from torch.utils.data import Dataset, DataLoader
import os
import json
from transformers import get_linear_schedule_with_warmup
from transformers import AdamW
from eval.syllabator import dashed_syllabified_line, syllabify

import argparse


parser = argparse.ArgumentParser()

parser.add_argument("--batch_size", default=32, type=int, help="Batch size")
parser.add_argument("--model", default="GPT2_oscar", type=str, help="tinyLlama # GPT2_oscar # GPT2_czech_XL # Mistral_czech # ") 
parser.add_argument("--epochs", default=5, type=int, help="Number of epochs")
parser.add_argument("--starting_epoch", default=0, type=int, help="epoch to load, 0 for not loading")
parser.add_argument("--learning_rate", default=3e-5, type=float, help="Learning rate")
parser.add_argument("--warmup_steps", default=500, type=int, help="Warmup steps")
parser.add_argument("--dataset_path", default="./", type=str, help="./ # DATA/Velky_zpevnik # ")
parser.add_argument("--dataset_type", default=11, type=int, help="Dataset type: BASELINE = 1, SYLLABLES = 2, END_OF_LINES = 3, CHARACTERISTIC_WORDS = 4, UNRHYMED = 5, SYLLABLES_AND_WORDS = 6, SYLLABLES_AND_ENDS = 7, ENDS_AND_WORDS = 8")


args = parser.parse_args([] if "__file__" not in globals() else None)

BATCH_SIZE = args.batch_size
MODEL_NAME = args.model
EPOCHS = args.epochs
LEARNING_RATE = args.learning_rate
WARMUP_STEPS = args.warmup_steps
DATASET_PATH = args.dataset_path
DATASET_TYPE = DatasetType(args.dataset_type)

print(f"----------------- {MODEL_NAME} ------ {DATASET_TYPE} ----------------")

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
    model = AutoModelForCausalLM.from_pretrained("simecek/cswikimistral_0.1",
                                                 load_in_4bit=True,
                                                 torch_dtype=torch.float16)
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

        self.lines_list = []
        
        with open(lyrics_path, "r", encoding="utf-8") as json_file:
            dataset_dict = json.load(json_file)

        if dataset_type == DatasetType.BASELINE:
            for dat_i in dataset_dict:
                for line in dataset_dict[dat_i]['lyrics']:
                    self.lines_list.append(line)

        elif dataset_type == DatasetType.SYLLABLES:
            for dat_i in dataset_dict:
                for lin_i in range(len(dataset_dict[dat_i]['lyrics'])):
                    self.lines_list.append(f"{dataset_dict[dat_i]['syllables'][lin_i]} # {dataset_dict[dat_i]['lyrics'][lin_i]}")
                    
        elif dataset_type == DatasetType.END_OF_LINES:
            for dat_i in dataset_dict:
                for lin_i in range(len(dataset_dict[dat_i]['lyrics'])):
                    self.lines_list.append(f"{dataset_dict[dat_i]['line_endings'][lin_i]} # {dataset_dict[dat_i]['lyrics'][lin_i]}")
                
        elif dataset_type == DatasetType.CHARACTERISTIC_WORDS:
            for dat_i in dataset_dict:
                for lin_i in range(len(dataset_dict[dat_i]['lyrics'])):
                    if dataset_dict[dat_i]['line_keywords'][lin_i] == '':
                        continue
                    self.lines_list.append(f"{dataset_dict[dat_i]['line_keywords'][lin_i]} # {dataset_dict[dat_i]['lyrics'][lin_i]}")

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

                    self.lines_list.append(f"{unrhymed_len} # {unrhymed} #\n{dataset_dict[dat_i]['syllables'][lin_i]} # {original}")

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

                    self.lines_list.append(f"{unrhymed_len} # {unrhymed_end} # {unrhymed} #\n{dataset_dict[dat_i]['syllables'][lin_i]} # {dataset_dict[dat_i]['line_endings'][lin_i]} # {original}")

        elif dataset_type == DatasetType.SYLLABLES_AND_WORDS:
            for dat_i in dataset_dict:
                for lin_i in range(len(dataset_dict[dat_i]['lyrics'])):
                    if dataset_dict[dat_i]['line_keywords'][lin_i] == '':
                        continue
                    self.lines_list.append(f"{dataset_dict[dat_i]['line_keywords'][lin_i]} #\n{dataset_dict[dat_i]['syllables'][lin_i]} # {dataset_dict[dat_i]['lyrics'][lin_i]}")

        elif dataset_type == DatasetType.SYLLABLES_AND_ENDS:
            for dat_i in dataset_dict:
                for lin_i in range(len(dataset_dict[dat_i]['lyrics'])):
                    self.lines_list.append(f"{dataset_dict[dat_i]['syllables'][lin_i]} # {dataset_dict[dat_i]['line_endings'][lin_i]} # {dataset_dict[dat_i]['lyrics'][lin_i]}")

        elif dataset_type == DatasetType.ENDS_AND_WORDS:
            for dat_i in dataset_dict:
                for lin_i in range(len(dataset_dict[dat_i]['lyrics'])):
                    if dataset_dict[dat_i]['line_keywords'][lin_i] == '':
                        continue
                    self.lines_list.append(f"{dataset_dict[dat_i]['line_keywords'][lin_i]} #\n{dataset_dict[dat_i]['line_endings'][lin_i]} # {dataset_dict[dat_i]['lyrics'][lin_i]}")

        elif DATASET_TYPE == DatasetType.FORCED_SYLLABLES:
            for dat_i in dataset_dict:
                for lin_i in range(len(dataset_dict[dat_i]['lyrics'])):
                    self.lines_list.append(f"{dataset_dict[dat_i]['syllables'][lin_i]} # {dashed_syllabified_line(dataset_dict[dat_i]['lyrics'][lin_i])}")

        elif DATASET_TYPE == DatasetType.SYLLABLES_ENDS_WORDS:
            for dat_i in dataset_dict:
                for lin_i in range(len(dataset_dict[dat_i]['lyrics'])):
                    if dataset_dict[dat_i]['line_keywords'][lin_i] == '':
                        continue
                    self.lines_list.append(f"{dataset_dict[dat_i]['line_keywords'][lin_i]} #\n{dataset_dict[dat_i]['syllables'][lin_i]} # {dataset_dict[dat_i]['line_endings'][lin_i]} # {dataset_dict[dat_i]['lyrics'][lin_i]}")

        else:
            raise Exception(f"We don't support a Dataset type {dataset_type}")
        
    def __len__(self):
        return len(self.lines_list)

    def __getitem__(self, item):
        return self.lines_list[item]

dataset = LyricsDataset()
lyrics_loader = DataLoader(dataset, batch_size=1, shuffle=True)

if args.starting_epoch != 0:
    model.load_state_dict(state_dict=torch.load(os.path.join(DATASET_PATH, "trained_models", f"{MODEL_NAME}_{DATASET_TYPE.name}_one_line_lyricist_{args.starting_epoch - 1}.pt"), map_location=torch.device(device)))

model.train()
optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps = WARMUP_STEPS, num_training_steps = -1)
proc_seq_count = 0
sum_loss = 0.0
batch_count = 0

tmp_lyrics_tens = None
models_folder = "trained_models"
if not os.path.exists(os.path.join(DATASET_PATH, models_folder)):
    os.mkdir(os.path.join(DATASET_PATH, models_folder))

for epoch in range(args.starting_epoch, EPOCHS):
    
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
            torch.save(model.state_dict(), os.path.join(DATASET_PATH, models_folder, f"{MODEL_NAME}_{DATASET_TYPE.name}_one_line_lyricist_{epoch}.pt"))
    
    # Store the model after each epoch to compare the performance of them
    torch.save(model.state_dict(), os.path.join(DATASET_PATH, models_folder, f"{MODEL_NAME}_{DATASET_TYPE.name}_one_line_lyricist_{epoch}.pt"))

