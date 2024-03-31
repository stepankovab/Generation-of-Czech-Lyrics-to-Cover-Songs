import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from dataset_types import DatasetType
from torch.utils.data import DataLoader
import os
from lyrics_datasets import LinesLyricsDataset, WholeLyricsDataset
from transformers import get_linear_schedule_with_warmup
from torch.optim import AdamW
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--batch_size", default=64, type=int, help="Batch size")
parser.add_argument("--model", default="GPT2_oscar", type=str, help="tinyLlama # GPT2_oscar # GPT2_czech_XL # Mistral_czech # ") 
parser.add_argument("--model_path", default="CODE/LLMExperiments/trained_models", type=str, help="./trained_models # ./ #  CODE/LLMExperiments/trained_models")
parser.add_argument("--epochs", default=5, type=int, help="Number of epochs")
parser.add_argument("--starting_epoch", default=0, type=int, help="epoch to load, 0 for not loading")
parser.add_argument("--learning_rate", default=5e-4, type=float, help="Learning rate")
parser.add_argument("--warmup_steps", default=200, type=int, help="Warmup steps")
parser.add_argument("--generation_method", default="whole", type=str, help="whole, lines")
parser.add_argument("--dataset_path", default="./", type=str, help="./ # DATA/Velky_zpevnik # ")
parser.add_argument("--dataset_type", default=12, type=int, help="""Dataset type: 
    BASELINE = 0
    SYLLABLES = 1
    SYLLABLES_ENDS = 2
    WORDS = 3
    WORDS_ENDS = 4
    SYLLABLES_WORDS = 5
    SYLLABLES_WORDS_ENDS = 6
    UNRHYMED_LEN = 7
    UNRHYMED_LEN_END = 8
    FORCED_SYLLABLES = 9
    FORCED_SYLLABLES_ENDS = 10
    RHYME_SCHEME = 11
    SYLLABLES_RHYME_SCHEME = 12
    SYLLABLES_RHYME_SCHEME_WORDS = 13""")
args = parser.parse_args([] if "__file__" not in globals() else None)

try:
    DATASET_TYPE = DatasetType(args.dataset_type)
except:
    raise ValueError(f"""{args.dataset_type} does not map onto any Dataset type.\n\nBASELINE = 0
    SYLLABLES = 1
    SYLLABLES_ENDS = 2
    WORDS = 3
    WORDS_ENDS = 4
    SYLLABLES_WORDS = 5
    SYLLABLES_WORDS_ENDS = 6
    UNRHYMED_LEN = 7
    UNRHYMED_LEN_END = 8
    FORCED_SYLLABLES = 9
    FORCED_SYLLABLES_ENDS = 10
    RHYME_SCHEME = 11
    SYLLABLES_RHYME_SCHEME = 12
    SYLLABLES_RHYME_SCHEME_WORDS = 13""")


if args.generation_method == "lines":
    dataset = LinesLyricsDataset(lyrics_dataset_path=args.dataset_path, dataset_type=DATASET_TYPE)
elif args.generation_method == "whole":
    dataset = WholeLyricsDataset(lyrics_dataset_path=args.dataset_path, dataset_type=DATASET_TYPE)
else:
    raise ValueError(f"Unsupported method: {args.generation_method}")

lyrics_loader = DataLoader(dataset, batch_size=1, shuffle=True)

print(f"----------------- {args.model} ------ {DATASET_TYPE.name} ----------------")

device = 'cpu'
if torch.cuda.is_available():
    print("cuda available.")
    device = 'cuda'
    torch.cuda.empty_cache()

if args.model == "GPT2_oscar":
    args.model = "lchaloupsky/czech-gpt2-oscar"
elif args.model == "GPT2_czech_XL":
    args.model = "BUT-FIT/Czech-GPT-2-XL-133k"
elif args.model == "tinyLlama":
    args.model = "TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T"

model, tokenizer = AutoModelForCausalLM.from_pretrained(args.model), AutoTokenizer.from_pretrained(args.model)

# Set special tokens if they are not already set
if tokenizer.sep_token is None:
    tokenizer.add_special_tokens({'sep_token': '[SEP]'})
if tokenizer.cls_token is None:
    tokenizer.add_special_tokens({'cls_token': '[CLS]'})
if tokenizer.mask_token is None:
    tokenizer.add_special_tokens({'mask_token': '[MASK]'})
    
model = model.to(device)

if args.starting_epoch != 0:
    model.load_state_dict(state_dict=torch.load(os.path.join(args.model_path, f"{args.model}_{DATASET_TYPE.name}_{args.generation_method}_{args.starting_epoch - 1}.pt"), map_location=torch.device(device)))

model.train()
optimizer = AdamW(model.parameters(), lr=args.learning_rate)
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps = args.warmup_steps, num_training_steps = args.epochs * (len(lyrics_loader) // args.batch_size))
sum_loss = 0.0
batch_count = 0

if not os.path.exists(args.model_path):
    os.mkdir(args.model_path)

for epoch in range(args.starting_epoch, args.epochs):
    print('=' * 30 + f" EPOCH {epoch} started " + '=' * 30)
    for idx, lyrics in enumerate(lyrics_loader):
        lyric_tens = torch.tensor(tokenizer.encode(lyrics[0])).unsqueeze(0).to(device)

        outputs = model(lyric_tens, labels=lyric_tens)
        loss, logits = outputs[:2] 
        sum_loss += loss.detach().data
                     
        loss.backward()
        if (idx + 1) % args.batch_size == 0:
            optimizer.step()
            scheduler.step() 
            optimizer.zero_grad()
            model.zero_grad()

        if (idx + 1) % (args.batch_size * 100) == 0:
            print(f"Epoch {epoch}, Average Loss: {sum_loss:.4f}")
            sum_loss = 0.0
            torch.save(model.state_dict(), os.path.join(args.model_path, f"{args.model}_{DATASET_TYPE.name}_{args.generation_method}_{epoch}.pt"))
    
    torch.save(model.state_dict(), os.path.join(args.model_path, f"{args.model}_{DATASET_TYPE.name}_{args.generation_method}_{epoch}.pt"))
