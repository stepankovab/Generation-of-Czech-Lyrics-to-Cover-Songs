import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel

MODEL_PATH = 'CODE/GPT2Experiments/tuned_models/gpt2_medium_joker_4.pt'

device = 'cpu'
if torch.cuda.is_available():
    print("cuda available.")
    device = 'cuda'

tokenizer = GPT2Tokenizer.from_pretrained("lchaloupsky/czech-gpt2-oscar")
tokenizer.model_max_length=1024


model = GPT2LMHeadModel.from_pretrained("lchaloupsky/czech-gpt2-oscar")
model.load_state_dict(state_dict=torch.load(MODEL_PATH, map_location=torch.device(device)))
model.eval()


# input sequence
text = "Slepě mlátí do piána"
inputs = tokenizer(text, return_tensors="pt") # tokenizer.encode(text, return_tensors="pt") directly for input_ids

# model output using Top-k sampling text generation method
sample_outputs = model.generate(inputs.input_ids,
                                pad_token_id=50256,
                                do_sample=True, 
                                max_length=50, # put the token number you want
                                top_k=40,
                                num_return_sequences=6)

# generated sequence
for i, sample_output in enumerate(sample_outputs):
    print("{}\n\n{}".format(i+1, tokenizer.decode(sample_output.tolist()))) # tokenizer.decode(sample_output, skip_special_tokens=True)



####################################### results #########################################
    
    # Nemám vůli žít, život brát si pro něj, to přece není fér, žít, žít si svět tak, jak nás baví<|endoftext|>
    # <|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|>
    # <|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|>
    # <|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|>oved<|endoftext|><|endoftext|>

    # Univerzita je základem rodiny, bez rodiny nic není, i když třeba pár věcí bolí, to se v každé rodině stává, 
    # proto ať má člověk to štěstí, že může žít<|endoftext|>, v páru<|endoftext|><|endoftext|><|endoftext|><|endoftext|>
    # <|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|>

    # Univerzita je základem poznání, když se mluví o kráse, pak i o lásce, kdopak ví<|endoftext|><|endoftext|>
    # <|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|>
    # <|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endext|><|endoftext|><|endoftext|><|endoftext|>
    # <|endoftext|><|endoftext|>

    # Univerzita je základem celé rodiny, ale hlavně v ní, nikdo neví, jaká jsou pravidla a kdo je ovládá, nikdo neví, 
    # jaký je to mít s někým soucit<|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|>
    # <|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|>
    # <|endoftext|><|endoftext|><|endoftext|>

