import torch
from utils.validators import *
from utils.poet_utils import *
from transformers import AutoTokenizer

model : ValidatorInterface  = torch.load("C:/Users/barca/MOJE/BAKALARKA/CODE/ScoreFunction/Imports/BPE_validator_1697833311028", map_location=torch.device('cpu'))

tokenizer = AutoTokenizer.from_pretrained('roberta-base')

ids = tokenizer("Když měla jsem svatbu mít")

print(model.validate(input_ids=ids)) #input_ids=ids, metre=TextAnalysis._metre_vector("T")








