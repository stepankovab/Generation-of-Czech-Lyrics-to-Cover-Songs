import torch
from utils.validators import *
from utils.poet_utils import *
from transformers import AutoTokenizer

model : ValidatorInterface  = torch.load("C:/Users/barca/MOJE/BAKALARKA/CODE/ScoreFunction/Imports/BPE_validator_1697833311028", map_location=torch.device('cpu'))

tokenizer = AutoTokenizer.from_pretrained('roberta-base')

ids = tokenizer("Když měla jsem svatbu mít", return_tensors='pt', truncation=True, padding=True).data["input_ids"]

print("J", model.validate(input_ids=ids, metre=torch.Tensor(TextAnalysis._metre_vector("J"))))
print("T", model.validate(input_ids=ids, metre=torch.Tensor(TextAnalysis._metre_vector("T"))))
print("D", model.validate(input_ids=ids, metre=torch.Tensor(TextAnalysis._metre_vector("D"))))
print("A", model.validate(input_ids=ids, metre=torch.Tensor(TextAnalysis._metre_vector("A"))))
print("X", model.validate(input_ids=ids, metre=torch.Tensor(TextAnalysis._metre_vector("X"))))
print("Y", model.validate(input_ids=ids, metre=torch.Tensor(TextAnalysis._metre_vector("Y"))))
print("N", model.validate(input_ids=ids, metre=torch.Tensor(TextAnalysis._metre_vector("N"))))
print("H", model.validate(input_ids=ids, metre=torch.Tensor(TextAnalysis._metre_vector("H"))))
print("P", model.validate(input_ids=ids, metre=torch.Tensor(TextAnalysis._metre_vector("P"))))

