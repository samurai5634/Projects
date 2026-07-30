import torch
import torch.nn as nn

class FENCF(nn.Module):
    def __init__(self, num_users, num_items, num_genres, factor_num=32, layers=[64, 32, 16]):
        super(FENCF, self).__init__()
        
        # -------------------------------------------------------------
        # 1. GMF Branch (Generalized Matrix Factorization - Linear)
        # -------------------------------------------------------------
        self.embed_user_GMF = nn.Embedding(num_users, factor_num)
        self.embed_item_GMF = nn.Embedding(num_items, factor_num)
        
        # -------------------------------------------------------------
        # 2. MLP Branch (Multi-Layer Perceptron - Non-linear)
        # -------------------------------------------------------------
        # User & Item ID Embeddings
        self.embed_user_MLP = nn.Embedding(num_users, layers[0] // 2)
        self.embed_item_MLP = nn.Embedding(num_items, layers[0] // 2)
        
        # Demographic Metadata Embeddings
        self.embed_gender = nn.Embedding(2, 4)      # 2 choices (0/1) -> 4-dim vector
        self.embed_age = nn.Embedding(60, 4)         # 7 age buckets -> 4-dim vector
        self.embed_occ = nn.Embedding(21, 8)        # 21 occupations -> 8-dim vector
        
        # Combine MLP input sizes: User_MLP + Item_MLP + Gender + Age + Occupation
        mlp_input_size = layers[0] + 4 + 4 + 8 
        
        # Build Dense Layers dynamically
        self.mlp_layers = nn.ModuleList()
        curr_size = mlp_input_size
        for out_size in layers[1:]:
            self.mlp_layers.append(nn.Linear(curr_size, out_size))
            self.mlp_layers.append(nn.ReLU())
            self.mlp_layers.append(nn.Dropout(p=0.2)) # Dropout to prevent overfitting
            curr_size = out_size

        # -------------------------------------------------------------
        # 3. Feature Enhancement (Genre Processing)
        # -------------------------------------------------------------
        # Projects 18-dim multi-hot binary vector into a learned dense representation
        self.genre_fc = nn.Linear(num_genres, 16) 

        # -------------------------------------------------------------
        # 4. Final Prediction Layer (Fusion)
        # -------------------------------------------------------------
        # Concatenates: GMF vector (32) + MLP output (16) + Genre vector (16)
        final_input_size = factor_num + layers[-1] + 16
        self.prediction_layer = nn.Linear(final_input_size, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, u_idx, i_idx, gender, age, occ, genres):
        # --- GMF Pathway ---
        user_gmf = self.embed_user_GMF(u_idx)
        item_gmf = self.embed_item_GMF(i_idx)
        gmf_out = user_gmf * item_gmf  # Element-wise product (Linear interaction)
        
        # --- MLP Pathway ---
        user_mlp = self.embed_user_MLP(u_idx)
        item_mlp = self.embed_item_MLP(i_idx)
        gen_feat = self.embed_gender(gender)
        age_feat = self.embed_age(age)
        occ_feat = self.embed_occ(occ)
        
        # Concatenate IDs + User Metadata
        mlp_in = torch.cat([user_mlp, item_mlp, gen_feat, age_feat, occ_feat], dim=-1)
        for layer in self.mlp_layers:
            mlp_in = layer(mlp_in)
            
        # --- Genre Pathway ---
        genre_out = torch.relu(self.genre_fc(genres))
        
        # --- Fusion Layer ---
        combined = torch.cat([gmf_out, mlp_in, genre_out], dim=-1)
        
        # Output probability [0.0, 1.0]
        prediction = self.prediction_layer(combined)
        return self.sigmoid(prediction).view(-1)