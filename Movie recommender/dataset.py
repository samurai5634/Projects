import torch
from torch.utils.data import Dataset

class MovieLensDataset(Dataset):
    def __init__(self, user_ids, item_ids, labels, user_meta_df, item_genre_matrix):
        """
        user_ids: Array of user indices
        item_ids: Array of item indices
        labels: Array of 1s and 0s
        user_meta_df: DataFrame indexed by user_id containing ['gender', 'age', 'occupation']
        item_genre_matrix: NumPy array / Tensor of shape (num_items, num_genres)
        """
        self.user_ids = torch.tensor(user_ids, dtype=torch.long)
        self.item_ids = torch.tensor(item_ids, dtype=torch.long)
        self.labels = torch.tensor(labels, dtype=torch.float32)
        
        # User Metadata lookups
        self.genders = torch.tensor(user_meta_df['gender'].values, dtype=torch.long)
        self.ages = torch.tensor(user_meta_df['age'].values, dtype=torch.long)
        self.occupations = torch.tensor(user_meta_df['occupation'].values, dtype=torch.long)
        
        # Movie Multi-Hot Genres matrix
        self.genres = torch.tensor(item_genre_matrix, dtype=torch.float32)

    def __len__(self):
        return len(self.user_ids)

    def __getitem__(self, idx):
        u = self.user_ids[idx]
        i = self.item_ids[idx]
        
        return {
            'user': u,
            'item': i,
            'label': self.labels[idx],
            'gender': self.genders[u],       # Index into user metadata
            'age': self.ages[u],
            'occupation': self.occupations[u],
            'genre': self.genres[i]          # Index into multi-hot genre matrix
        }