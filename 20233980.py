import os
import random
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, GPT2Config, GPT2Model

from base import eli5_dataset, set_seed

import os
import random
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, GPT2Config, GPT2Model

from base import eli5_dataset, set_seed


class GPTLanguageModel(nn.Module):
    def __init__(self, vocab_size, d_model=768, nhead=12, num_layers=6, d_ff=3072, max_seq_len=200, dropout=0.1):
        super().__init__()
        
        # Custom config to match expected dimensions
        custom_config = GPT2Config(
            vocab_size=vocab_size,
            n_embd=d_model,          # Hidden size 
            n_head=nhead,            # Number of attention heads
            n_layer=num_layers,      # Number of decoder layers
            n_inner=d_ff,            # FFN inner hidden size
            n_positions=max_seq_len, # Explicitly set to 200
            resid_pdrop=dropout,     # Residual dropout
            embd_pdrop=dropout,      # Embedding dropout
            attn_pdrop=dropout,      # Attention dropout
            use_cache=False          # No need for past key values during training
        )
        
        # Initialize GPT-2 model 
        self.transformer = GPT2Model.from_pretrained(
            "gpt2", 
            config=custom_config,
            ignore_mismatched_sizes=True
        )
        
        # Modify embedding layers to match expected dimensions
        # Embedding for tokens
        self.transformer.wte = nn.Embedding(vocab_size, d_model)
        
        # Positional embedding
        self.transformer.wpe = nn.Embedding(max_seq_len, d_model)
        
        # Adjust layer dimensions
        for layer in self.transformer.h:
            # Adjust layer norm dimensions
            layer.ln_1 = nn.LayerNorm(d_model)
            layer.ln_2 = nn.LayerNorm(d_model)
            
            # Adjust attention layer
            layer.attn.c_attn = nn.Linear(d_model, d_model * 3)
            layer.attn.c_proj = nn.Linear(d_model, d_model)
            
            # Adjust MLP layers
            layer.mlp.c_fc = nn.Linear(d_model, d_ff)
            layer.mlp.c_proj = nn.Linear(d_ff, d_model)
        
        # Final layer norm
        self.transformer.ln_f = nn.LayerNorm(d_model)
        
        # Output layer
        self.fc_out = nn.Linear(d_model, vocab_size)
        
    def forward(self, x):
        # Ensure input matches expected shape
        transformer_outputs = self.transformer(x)
        hidden_states = transformer_outputs.last_hidden_state
        logits = self.fc_out(hidden_states)
        return logits

def evaluate_model(model, dataloader, device):
    model.eval()
    # Preallocate numpy array with exact required shape
    all_logits = np.zeros((75, 200, 50257), dtype=np.float32)
    
    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            if i >= 75:  # Ensure we only process the exact number of test samples
                break
            
            batch = batch.to(device)
            # Generate logits for all positions
            logits = model(batch)
            #print('logits shape:',logits.shape)
            
            # Ensure we are filling the preallocated array with the correct slicing
            end_idx = min(len(batch), 75 - i * dataloader.batch_size)
            all_logits[i * dataloader.batch_size:i * dataloader.batch_size + end_idx] = logits.cpu().numpy()[:end_idx]
    
    return all_logits



def calculate_loss(outputs, targets, criterion):
    # Reshape outputs and targets for loss calculation
    outputs = outputs.view(-1, outputs.size(-1))
    targets = targets.reshape(-1)
    return criterion(outputs, targets)

def train_model(model, dataloader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    for batch in dataloader:
        batch = batch.to(device)
        optimizer.zero_grad()
        outputs = model(batch[:, :-1])
        loss = calculate_loss(outputs, batch[:, 1:], criterion)
        loss.backward()
        # Gradient clipping to prevent exploding gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(dataloader)

def validate_model(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for batch in dataloader:
            batch = batch.to(device)
            outputs = model(batch[:, :-1])
            loss = calculate_loss(outputs, batch[:, 1:], criterion)
            total_loss += loss.item()
    return total_loss / len(dataloader)


def main():
    # Set seed for reproducibility
    set_seed(0)
    
    # Parameters
    BATCH_SIZE = 32
    MAX_POSITION_EMBEDDINGS = 200
    LEARNING_RATE = 1e-4
    NUM_EPOCHS = 50
    VOCAB_SIZE = 50257  # GPT-2 tokenizer vocab size
    
    # Model hyperparameters
    D_MODEL = 768 #512       # Hidden size
    N_HEAD = 12 #8         # Number of attention heads
    N_LAYER = 6        # Number of decoder layers
    D_FF = 3072 #2048        # FFN inner hidden size
    DROPOUT = 0.1      # Dropout rate
    
    # Device configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load and configure tokenizer
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    # Set pad token to be equal to the EOS token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    
    # Load datasets
    trainset = eli5_dataset(tokenizer, MAX_POSITION_EMBEDDINGS, "train")
    validset = eli5_dataset(tokenizer, MAX_POSITION_EMBEDDINGS, "valid")
    testset = eli5_dataset(tokenizer, MAX_POSITION_EMBEDDINGS, "test")
    
    train_loader = DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True)
    valid_loader = DataLoader(validset, batch_size=BATCH_SIZE)
    test_loader = DataLoader(testset, batch_size=BATCH_SIZE)
    
    print(f"Train size: {len(trainset)}, Valid size: {len(validset)}, Test size: {len(testset)}")
    
    # Initialize model
    model = GPTLanguageModel(
        vocab_size=VOCAB_SIZE,
        d_model=D_MODEL,
        nhead=N_HEAD,
        num_layers=N_LAYER,
        d_ff=D_FF,
        max_seq_len=MAX_POSITION_EMBEDDINGS,
        dropout=DROPOUT
    ).to(device)
    
    # Initialize optimizer with weight decay
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=0.01)
    
    # Loss function with explicitly set pad_token_id
    criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)
    
    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=LEARNING_RATE,
        steps_per_epoch=len(train_loader),
        epochs=NUM_EPOCHS,
        pct_start=0.1
    )
    
    # Training loop
    best_val_loss = float('inf')
    for epoch in range(NUM_EPOCHS):
        # Training
        train_loss = train_model(model, train_loader, optimizer, criterion, device)
        
        # Validation
        val_loss = validate_model(model, valid_loader, criterion, device)
        
        # Update learning rate
        scheduler.step()
        
        # Print progress
        print(f"Epoch {epoch + 1}/{NUM_EPOCHS}")
        print(f"Training Loss: {train_loss:.4f}")
        print(f"Validation Loss: {val_loss:.4f}")
        print(f"Learning Rate: {scheduler.get_last_lr()[0]:.6f}")
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), "best_model.pt")
    
    # Load best model for final evaluation
    model.load_state_dict(torch.load("best_model.pt"))
    
    # Generate and save logits
    logits = evaluate_model(model, test_loader, device)
    np.save("20233980.npy", logits)  # Replace with your student ID
    print("Logits saved successfully!")

if __name__ == "__main__":
    main()