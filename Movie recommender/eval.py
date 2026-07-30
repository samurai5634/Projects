import pickle
import torch
from model import FENCF
from metrics import evaluate_model

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Executing evaluation on device: {device}")

    # 1. Load Preprocessed Data Artifacts
    print("Loading preprocessed artifacts...")
    with open("data/movielens_preprocessed.pkl", "rb") as f:
        artifacts = pickle.load(f)

    train_ratings = artifacts['train_ratings']
    test_ratings = artifacts['test_ratings']
    user_meta_df = artifacts['user_meta_df']
    item_genre_matrix = artifacts['item_genre_matrix']

    NUM_USERS = artifacts['num_users']
    NUM_ITEMS = artifacts['num_items']
    NUM_GENRES = artifacts['num_genres']

    # 2. Re-instantiate the Model Architecture
    print("Instantiating FENCF model...")
    model = FENCF(
        num_users=NUM_USERS,
        num_items=NUM_ITEMS,
        num_genres=NUM_GENRES,
        factor_num=32,
        layers=[64, 32, 16]
    ).to(device)

    # 3. Load Saved Weights
    checkpoint_path = "checkpoints/best_fencf_model.pth"
    print(f"Loading checkpoint weights from: {checkpoint_path}")
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))

    # 4. Execute Evaluation Protocol
    print("\n--- Starting Leave-One-Out Evaluation (HR@10 & NDCG@10) ---")
    hr, ndcg = evaluate_model(
        model=model,
        test_ratings=test_ratings,
        train_ratings=train_ratings,
        user_meta_df=user_meta_df,
        item_genre_matrix=item_genre_matrix,
        num_items=NUM_ITEMS,
    )

    print("\n================ EVALUATION RESULTS ================")
    print(f"Hit Ratio @ 10 (HR@10):   {hr:.4f}")
    print(f"NDCG @ 10      (NDCG@10): {ndcg:.4f}")
    print("====================================================")

if __name__ == "__main__":
    main()

# ================ EVALUATION RESULTS ================
# Hit Ratio @ 10 (HR@10):   0.6674
# NDCG @ 10      (NDCG@10): 0.3853
# ====================================================