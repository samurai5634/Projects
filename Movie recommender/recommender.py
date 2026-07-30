import torch
import numpy as np

# MovieLens 1M Genre List in exact index order
GENRES = [
    "Action", "Adventure", "Animation", "Children's", "Comedy", "Crime",
    "Documentary", "Drama", "Fantasy", "Film-Noir", "Horror", "Musical",
    "Mystery", "Romance", "Sci-Fi", "Thriller", "War", "Western"
]

def get_genres_from_vector(genre_vector):
    """Convert one-hot binary genre vector into readable genre strings."""
    return [GENRES[i] for i, val in enumerate(genre_vector) if val == 1]

@torch.no_grad()
def get_user_watched_history(user_id, train_ratings, item_genre_matrix, idx_to_title, top_n=6):
    """Fetch recent watched movies with titles and genres."""
    user_ratings = train_ratings[train_ratings['user_id'] == user_id].head(top_n)
    
    watched_items = []
    for _, row in user_ratings.iterrows():
        item_id = int(row['item_id'])
        title = idx_to_title.get(item_id, "Unknown Movie")
        genres = get_genres_from_vector(item_genre_matrix[item_id])
        watched_items.append({
            "title": title,
            "genres": genres[:2] # Grab top 2 genres for compact display
        })
        
    return watched_items

@torch.no_grad()
def recommend_for_user(user_id, model, train_ratings, user_meta_df, item_genre_matrix, idx_to_title, num_items, top_k=6, device="cpu"):
    """Generate Top-K visual movie cards for unseen items."""
    model.eval()
    
    seen_items = set(train_ratings[train_ratings['user_id'] == user_id]['item_id'].unique())
    candidate_items = list(set(range(num_items)) - seen_items)
    num_candidates = len(candidate_items)
    
    if num_candidates == 0:
        return []
    
    u_tensor = torch.tensor([user_id] * num_candidates, dtype=torch.long, device=device)
    i_tensor = torch.tensor(candidate_items, dtype=torch.long, device=device)
    
    gender_val = int(user_meta_df.loc[user_id, 'gender'])
    age_val = int(user_meta_df.loc[user_id, 'age'])
    occ_val = int(user_meta_df.loc[user_id, 'occupation'])
    
    gender_tensor = torch.tensor([gender_val] * num_candidates, dtype=torch.long, device=device)
    age_tensor = torch.tensor([age_val] * num_candidates, dtype=torch.long, device=device)
    occ_tensor = torch.tensor([occ_val] * num_candidates, dtype=torch.long, device=device)
    genre_tensor = torch.tensor(item_genre_matrix[candidate_items], dtype=torch.float32, device=device)
    
    predictions = model(u_tensor, i_tensor, gender_tensor, age_tensor, occ_tensor, genre_tensor)
    scores = predictions.view(-1).cpu().numpy()
    
    top_indices = np.argsort(scores)[::-1][:top_k]
    top_item_ids = [candidate_items[idx] for idx in top_indices]
    top_scores = [scores[idx] for idx in top_indices]
    
    recommendations = []
    for rank, (item_id, score) in enumerate(zip(top_item_ids, top_scores), 1):
        movie_title = idx_to_title.get(item_id, "Unknown Movie")
        genres = get_genres_from_vector(item_genre_matrix[item_id])
        recommendations.append({
            "rank": rank,
            "title": movie_title,
            "match": int(score * 100), # Convert to integer percentage e.g. 94%
            "genres": genres[:2]
        })
        
    return recommendations