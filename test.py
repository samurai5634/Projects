import pickle
import torch
from model import FENCF
from recommender import recommend_for_user, get_user_watched_history
from mappings import AGE_MAP, OCCUPATION_MAP, GENDER_MAP

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running inference on device: {device}\n")

    # 1. Load Preprocessed Data
    print("Loading data artifacts...")
    with open("data/movielens_preprocessed.pkl", "rb") as f:
        artifacts = pickle.load(f)

    train_ratings = artifacts['train_ratings']
    user_meta_df = artifacts['user_meta_df']
    item_genre_matrix = artifacts['item_genre_matrix']
    idx_to_title = artifacts['idx_to_title']
    num_items = artifacts['num_items']

    # 2. Instantiate & Load Model
    print("Loading model weights...")
    model = FENCF(
        num_users=artifacts['num_users'],
        num_items= num_items,
        num_genres=artifacts['num_genres'],
        factor_num=32,
        layers=[64, 32, 16]
    ).to(device)

    checkpoint_path = "checkpoints/best_fencf_model.pth"
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))

    # 3. Test for a specific user ID (e.g., user_id = 42)
    test_user_id = 42
    
    # Extract Demographics
    raw_gender = user_meta_df.loc[test_user_id, 'gender']
    raw_age = user_meta_df.loc[test_user_id, 'age']
    raw_occ = user_meta_df.loc[test_user_id, 'occupation']

    gender_str = GENDER_MAP.get(raw_gender, "Unknown")
    age_str = AGE_MAP.get(raw_age, "Unknown")
    occ_str = OCCUPATION_MAP.get(raw_occ, "Unknown")

    print("\n==================================================")
    print(f"USER PROFILE (Mapped ID: {test_user_id})")
    print(f"Demographics: {gender_str} | Age: {age_str} | Occupation: {occ_str}")
    print("==================================================")

    # 4. Show History
    print("\n--- Top Previously Watched Movies ---")
    history_df = get_user_watched_history(test_user_id, train_ratings, idx_to_title, top_n=5)
    print(history_df.to_string(index=False))

    # 5. Generate Recommendations
    print("\n--- Model Top-10 Recommendations ---")
    recs_df = recommend_for_user(
        user_id=test_user_id,
        model=model,
        train_ratings=train_ratings,
        user_meta_df=user_meta_df,
        item_genre_matrix=item_genre_matrix,
        idx_to_title=idx_to_title,
        num_items=num_items,
        top_k=10,
        device=device
    )
    print(recs_df.to_string(index=False))
    print("==================================================\n")

if __name__ == "__main__":
    main()