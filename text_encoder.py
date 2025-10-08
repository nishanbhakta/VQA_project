import torch
import torch.nn as nn
from transformers import BertModel, BertTokenizer

# Initialize tokenizer
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

class TextEncoder(nn.Module):
    def __init__(self, output_dim=768):
        super(TextEncoder, self).__init__()
        self.bert = BertModel.from_pretrained("bert-base-uncased")
        self.seq_proj = nn.Linear(self.bert.config.hidden_size, output_dim)
        self.proj = nn.Linear(self.bert.config.hidden_size, output_dim)  # optional projection

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        seq_emb = self.seq_proj(outputs.last_hidden_state)  # project each token
        return seq_emb

# Helper function to tokenize
def prepare_text(questions, device="cpu", max_length=32):
    enc = tokenizer(questions, padding="max_length", truncation=True, max_length=max_length, return_tensors="pt")
    return enc["input_ids"].to(device), enc["attention_mask"].to(device)

""" 
device = "cuda" if torch.cuda.is_available() else "cpu"
encoder = TextEncoder(output_dim=768).to(device)

# Example sentence
sentence = "How many cats are sitting on the sofa?"

# Step 1: Tokenize
input_ids, attention_mask = prepare_text([sentence], device=device, max_length=32)

# Step 2: Get embeddings
with torch.no_grad():
    seq_emb = encoder(input_ids, attention_mask)

print("Token embeddings shape:", seq_emb.shape)   # (batch_size, max_length, hidden_dim)
### print("Token embeddings: \n", seq_emb)
"""
