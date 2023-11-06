import torch
from Imports.validators import *
from Imports.poet_utils import *
from transformers import AutoTokenizer
from pathlib import Path

model_path = Path("C:\Users\barca\MOJE\BAKALARKA\CODE\ScoreFunction\Imports\funguje_vole")

model = MeterValidator(torch.load(model_path, map_location=torch.device('cpu')))
tokenizer =  AutoTokenizer.from_pretrained('roberta-base')

model.validate(input_ids=[1,2], metre=[3,4  ])['acc']





