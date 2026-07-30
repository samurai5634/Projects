import os
import pickle
import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from model import FENCF
from dataset import MovieLensDataset


# ==========================================
# 0. Loading data
# ==========================================

# 1. Load the preprocessed dictionary artifact from disk
print("Loading preprocessed MovieLens dataset...")
with open("data/movielens_preprocessed.pkl", "rb") as f:
    artifacts = pickle.load(f)

# 2. Extract DataFrames and Feature Matrices
train_ratings = artifacts['train_ratings']
test_ratings = artifacts['test_ratings']
user_meta_df = artifacts['user_meta_df']
item_genre_matrix = artifacts['item_genre_matrix']


# 3. Extract Model Initialization Dimensions
NUM_USERS = artifacts['num_users']     # 6040
NUM_ITEMS = artifacts['num_items']     # 3706
NUM_GENRES = artifacts['num_genres']   # 18

print(f"Data successfully loaded!")
print(f"Train samples: {len(train_ratings):,}, Test users: {len(test_ratings):,}")
print(f"Users: {NUM_USERS}, Items: {NUM_ITEMS}, Genres: {NUM_GENRES}")



# ==========================================
# 1. Reproducibility & Setup
# ==========================================
def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

seed_everything(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Executing on device: {device}")


# ==========================================
# 2. Dynamic Negative Sampling Helper
# ==========================================

def get_train_instances(train_ratings, num_items , num_negatives=4):
    user_input, item_input, labels = [], [], []
    # Create a set of all (user, item) pairs for fast lookup
    train_set = set(zip(train_ratings['user_id'], train_ratings['item_id']))
    
    for u, i in train_set:
        # 1. Add the positive instance
        user_input.append(u)
        item_input.append(i)
        labels.append(1)
        
        # 2. Add 'num_negatives' negative instances
        for _ in range(num_negatives):
            j = np.random.randint(num_items)
            while (u, j) in train_set:
                j = np.random.randint(num_items)
            user_input.append(u)
            item_input.append(j)
            labels.append(0)
            
    return np.array(user_input), np.array(item_input), np.array(labels)


# ==========================================
# 3. Training & Validation Epoch Loops
# ==========================================
def train_one_epoch(model, dataloader, optimizer, criterion, device):
    model.train()
    running_loss = 0.0

    for batch in dataloader:
        # Move batch inputs to active compute device
        u = batch['user'].to(device)
        i = batch['item'].to(device)
        y = batch['label'].to(device)
        gender = batch['gender'].to(device)
        age = batch['age'].to(device)
        occ = batch['occupation'].to(device)
        genre = batch['genre'].to(device)

        optimizer.zero_grad()
        
        # Forward Pass
        predictions = model(u, i, gender, age, occ, genre)
        loss = criterion(predictions.view(-1), y.float())

        # Backward Pass & Optimization
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * u.size(0)

    return running_loss / len(dataloader.dataset)


# ==========================================
# 4. Main Training Pipeline Execution
# ==========================================
def main():
    # --- Hyperparameters ---
    
    EPOCHS = 20
    BATCH_SIZE = 256
    LR = 0.001
    FACTOR_NUM = 32
    MLP_LAYERS = [64, 32, 16]

    
    print("Building FENCF Model Architecture...")
    model = FENCF(
        num_users=NUM_USERS,
        num_items=NUM_ITEMS,
        num_genres=NUM_GENRES,
        factor_num=FACTOR_NUM,
        layers=MLP_LAYERS
    ).to(device)

    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-6)

    best_loss = float('inf')
    os.makedirs("checkpoints", exist_ok=True)

    print("\n--- Starting Training Engine ---")
    for epoch in range(1, EPOCHS + 1):
        # Step A: Dynamic Negative Resampling every epoch
        train_users, train_items, train_labels = get_train_instances(
           train_ratings,NUM_ITEMS)

        # Step B: Instantiate fresh Dataset & DataLoader
        train_dataset = MovieLensDataset(
            user_ids=train_users,
            item_ids=train_items,
            labels=train_labels,
            user_meta_df=user_meta_df,
            item_genre_matrix=item_genre_matrix
        )

        train_loader = DataLoader(
            train_dataset, 
            batch_size=BATCH_SIZE, 
            shuffle=True, 
            num_workers=2, 
            pin_memory=True
        )

        # Step C: Execute Epoch Pass
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        print(f"Epoch [{epoch:02d}/{EPOCHS:02d}] | Train BCE Loss: {train_loss:.4f}")

        # Step D: Save Model Checkpoint
        if train_loss < best_loss:
            best_loss = train_loss
            checkpoint_path = "checkpoints/best_fencf_model.pth"
            torch.save(model.state_dict(), checkpoint_path)

    print(f"\nTraining Complete! Best model saved to: checkpoints/best_fencf_model.pth")

if __name__ == "__main__":
    main()