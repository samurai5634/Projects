import os
import pickle
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import optuna
from optuna.samplers import TPESampler

from model import FENCF
from dataset import MovieLensDataset
from metrics import evaluate_model
from train import get_train_instances, seed_everything

# -------------------------------------------------------------
# 1. LOAD DATA ONCE GLOBALLY (Avoids reading 15x from disk)
# -------------------------------------------------------------
def load_data():
    with open("data/movielens_preprocessed.pkl", "rb") as f:
        return pickle.load(f)

print("📦 Loading preprocessed artifacts into memory...")
artifacts = load_data()
train_ratings = artifacts['train_ratings']
test_ratings = artifacts['test_ratings']
user_meta_df = artifacts['user_meta_df']
item_genre_matrix = artifacts['item_genre_matrix']
num_users = artifacts['num_users']
num_items = artifacts['num_items']
num_genres = artifacts['num_genres']


def objective(trial):
    """Optuna trial function for hyperparameter search."""
    lr = trial.suggest_float("lr", 1e-4, 1e-2, log=True)
    factor_num = trial.suggest_categorical("factor_num", [16, 32, 64])
    weight_decay = trial.suggest_float("weight_decay", 1e-7, 1e-3, log=True)
    batch_size = trial.suggest_categorical("batch_size", [256, 512, 1024])
    
    mlp_layer_choice = trial.suggest_categorical("mlp_layers", ["small", "medium", "large"])
    layer_map = {
        "small": [32, 16, 8],
        "medium": [64, 32, 16],
        "large": [128, 64, 32]
    }
    mlp_layers = layer_map[mlp_layer_choice]

    # Reduced to 3 epochs for fast screening during tuning
    TUNE_EPOCHS = 3
    NUM_NEGATIVES = 4
    
    seed_everything(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = FENCF(
        num_users=num_users,
        num_items=num_items,
        num_genres=num_genres,
        factor_num=factor_num,
        layers=mlp_layers
    ).to(device)

    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    for epoch in range(1, TUNE_EPOCHS + 1):
        train_users, train_items, train_labels = get_train_instances(
            train_ratings, num_items, num_negatives=NUM_NEGATIVES
        )

        train_dataset = MovieLensDataset(
            user_ids=train_users,
            item_ids=train_items,
            labels=train_labels,
            user_meta_df=user_meta_df,
            item_genre_matrix=item_genre_matrix
        )

        # num_workers=0 prevents process spawning slowdowns during short tuning runs
        train_loader = DataLoader(
            train_dataset, 
            batch_size=batch_size, 
            shuffle=True, 
            num_workers=0, 
            pin_memory=True if device.type == 'cuda' else False
        )

        model.train()
        for batch in train_loader:
            u = batch['user'].to(device)
            i = batch['item'].to(device)
            y = batch['label'].to(device)
            gender = batch['gender'].to(device)
            age = batch['age'].to(device)
            occ = batch['occupation'].to(device)
            genre = batch['genre'].to(device)

            optimizer.zero_grad()
            preds = model(u, i, gender, age, occ, genre)
            loss = criterion(preds.view(-1), y.float())
            loss.backward()
            optimizer.step()

    # Evaluate NDCG@10
    hr, ndcg = evaluate_model(
        model=model,
        test_ratings=test_ratings,
        train_ratings=train_ratings,
        user_meta_df=user_meta_df,
        item_genre_matrix=item_genre_matrix,
        num_items=num_items,
        top_k=10,
        device=device
    )

    return ndcg


def main():
    print("==================================================")
    print("🚀 STARTING OPTUNA HYPERPARAMETER TUNING PIPELINE")
    print("==================================================\n")


    storage_url = "sqlite:///optuna.db"
    
    study = optuna.create_study(
        study_name="fencf_single_obj_v1",
        direction="maximize",
        storage=storage_url,
        load_if_exists=False,
        sampler=TPESampler(seed=42)
    )

    # 10 trials is sufficient to sample the parameter space effectively
    N_TRIALS = 10
    study.optimize(objective, n_trials=N_TRIALS, show_progress_bar=True)

    print("\n==================================================")
    print("🏆 OPTIMIZATION COMPLETE!")
    print(f"Best Trial Score (NDCG@10): {study.best_value:.4f}")
    print("==================================================")
    print("Best Hyperparameters Found:")
    for key, value in study.best_params.items():
        print(f" - {key}: {value}")
    print("==================================================\n")


if __name__ == "__main__":
    main()


# Best Hyperparameters Found:
#  - lr: 0.008706020878304856
#  - factor_num: 16
#  - weight_decay: 5.415244119402535e-07
#  - batch_size: 512
#  - mlp_layers: medium