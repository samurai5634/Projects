import numpy as np
import torch
import math

def get_hit_ratio(ranklist, target_item):
    """
    Checks if the ground-truth target item is in the top-K recommendations.
    """
    return 1 if target_item in ranklist else 0

def get_ndcg(ranklist, target_item):
    """
    Calculates NDCG@K for the top-K recommendations.
    Position 1 yields 1.0; lower ranks yield exponentially decaying scores.
    """
    if target_item in ranklist:
        index = ranklist.index(target_item)
        return 1.0 / math.log2(index + 2)  # index starts at 0, rank starts at 1
    return 0.0

@torch.no_grad()

def evaluate_model(model, test_ratings, train_ratings, user_meta_df, item_genre_matrix, num_items, top_k=10, device="cpu"):
    """
    Evaluates FENCF using Leave-One-Out evaluation with 100 negative items per user.
    """
    model.eval()
    hits, ndcgs = [], []
    
    # Pre-build user interaction sets for fast negative sampling lookup
    train_user_items = train_ratings.groupby('user_id')['item_id'].apply(set).to_dict()
    test_user_items = test_ratings.groupby('user_id')['item_id'].apply(set).to_dict()
    
    # Evaluate across test users
    for row in test_ratings.itertuples():
        u = int(row.user_id)
        pos_item = int(row.item_id)
        
        # 1. Sample 100 negative items not in user's train or test sets
        seen_items = train_user_items.get(u, set()).union(test_user_items.get(u, set()))
        neg_items = []
        while len(neg_items) < 100:
            j = np.random.randint(num_items)
            if j not in seen_items and j not in neg_items:
                neg_items.append(j)
                
        # Combine into candidate item list (1 positive + 100 negatives)
        candidate_items = [pos_item] + neg_items
        
        # 2. Build Tensors for Model Forward Pass
        u_tensor = torch.tensor([u] * 101, dtype=torch.long, device=device)
        i_tensor = torch.tensor(candidate_items, dtype=torch.long, device=device)
        
        # Look up metadata repeated 101 times
        gender_val = int(user_meta_df.loc[u, 'gender'])
        age_val = int(user_meta_df.loc[u, 'age'])
        occ_val = int(user_meta_df.loc[u, 'occupation'])
        
        gender_tensor = torch.tensor([gender_val] * 101, dtype=torch.long, device=device)
        age_tensor = torch.tensor([age_val] * 101, dtype=torch.long, device=device)
        occ_tensor = torch.tensor([occ_val] * 101, dtype=torch.long, device=device)
        
        # Genre vectors for all 101 candidates
        genre_tensor = torch.tensor(item_genre_matrix[candidate_items], dtype=torch.float32, device=device)
        
        # 3. Predict Scores
        predictions = model(u_tensor, i_tensor, gender_tensor, age_tensor, occ_tensor, genre_tensor)
        scores = predictions.view(-1).cpu().numpy()
        
        # 4. Map candidate items to scores and sort top K
        item_score_map = dict(zip(candidate_items, scores))
        ranklist = sorted(item_score_map, key=item_score_map.get, reverse=True)[:top_k]
        
        # 5. Compute Metrics
        hits.append(get_hit_ratio(ranklist, pos_item))
        ndcgs.append(get_ndcg(ranklist, pos_item))
        
    avg_hr = np.mean(hits)
    avg_ndcg = np.mean(ndcgs)
    
    return avg_hr, avg_ndcg