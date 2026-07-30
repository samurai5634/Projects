import streamlit as st
import pickle
import torch
import pandas as pd
from model import FENCF
from recommender import recommend_for_user, get_user_watched_history
from mappings import AGE_MAP, OCCUPATION_MAP, GENDER_MAP

# --- Page Config ---
st.set_page_config(
    page_title="CineMatch AI | Deep Recommendations",
    page_icon="🍿",
    layout="wide"
)

# --- Custom Styling for Movie Grid Cards ---
st.markdown("""
    <style>
    /* Global Styling */
    .stApp {
        background-color: #0F0F12;
        color: #FFFFFF;
    }

    /* Card Component */
    .movie-card {
        background: #18181C;
        border: 1px solid #282830;
        border-radius: 12px;
        padding: 1rem;
        margin-bottom: 1rem;
        min-height: 140px;
        display: flex;
        flex-direction: column;
        justify: space-between;
        transition: transform 0.2s ease, border-color 0.2s ease;
    }
    
    .movie-card:hover {
        border-color: #E50914;
        transform: translateY(-2px);
    }

    /* Rank Badge */
    .rank-badge {
        background: #E50914;
        color: white;
        font-weight: 800;
        font-size: 0.75rem;
        padding: 2px 8px;
        border-radius: 6px;
        width: fit-content;
    }

    /* Title Styling */
    .movie-title {
        font-size: 1rem;
        font-weight: 700;
        color: #FFFFFF;
        margin: 8px 0px 4px 0px;
        line-height: 1.3;
    }

    /* Match Score Bar */
    .match-score {
        color: #46D369; /* Netflix Match Green */
        font-weight: 700;
        font-size: 0.9rem;
    }

    /* Genre Tag */
    .genre-tag {
        background: #25252E;
        color: #A0A0B0;
        font-size: 0.75rem;
        padding: 2px 8px;
        border-radius: 4px;
        margin-right: 4px;
    }

    .card-footer {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-top: auto;
        padding-top: 8px;
    }
    </style>
""", unsafe_allow_html=True)

# --- Resource Caching ---
@st.cache_resource
def load_assets():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    with open("data/movielens_preprocessed.pkl", "rb") as f:
        artifacts = pickle.load(f)
        
    model = FENCF(
        num_users=artifacts['num_users'],
        num_items=artifacts['num_items'],
        num_genres=artifacts['num_genres'],
        factor_num=32,
        layers=[64, 32, 16]
    ).to(device)
    
    checkpoint_path = "checkpoints/best_fencf_model.pth"
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()
    
    return artifacts, model, device

artifacts, model, device = load_assets()

train_ratings = artifacts['train_ratings']
user_meta_df = artifacts['user_meta_df']
item_genre_matrix = artifacts['item_genre_matrix']
idx_to_title = artifacts['idx_to_title']
num_items = artifacts['num_items']

user_watch_counts = train_ratings.groupby('user_id').size().to_dict()

# Reverse Mappings
GENDER_REV = {v: k for k, v in GENDER_MAP.items()}
AGE_REV = {v: k for k, v in AGE_MAP.items()}
OCCUPATION_REV = {v: k for k, v in OCCUPATION_MAP.items()}

# --- Header ---
st.title("🍿 CineMatch AI Dashboard")

# --- Sidebar Controls ---
st.sidebar.markdown("### 🎯 Demographic Persona")
selected_gender_label = st.sidebar.selectbox("Gender", ["All"] + list(GENDER_MAP.values()))
selected_age_label = st.sidebar.selectbox("Age Group", ["All"] + list(AGE_MAP.values()))
selected_occ_label = st.sidebar.selectbox("Occupation", ["All"] + list(OCCUPATION_MAP.values()))

filtered_df = user_meta_df.copy()

if selected_gender_label != "All":
    filtered_df = filtered_df[filtered_df['gender'] == GENDER_REV[selected_gender_label]]
if selected_age_label != "All":
    filtered_df = filtered_df[filtered_df['age'] == AGE_REV[selected_age_label]]
if selected_occ_label != "All":
    filtered_df = filtered_df[filtered_df['occupation'] == OCCUPATION_REV[selected_occ_label]]

st.sidebar.markdown("---")

if filtered_df.empty:
    st.sidebar.error("No viewer profiles found matching criteria.")
    selected_user_id = None
else:
    user_options = {}
    for idx, uid in enumerate(filtered_df.index, 1):
        count = user_watch_counts.get(uid, 0)
        label = f"Viewer Persona #{idx} ({count} rated)"
        user_options[label] = uid

    selected_label = st.sidebar.selectbox(
        f"Select Profile ({len(user_options)} matching):", 
        list(user_options.keys())
    )
    selected_user_id = user_options[selected_label]

top_k = st.sidebar.slider("Number of Recommendations:", min_value=4, max_value=12, value=6, step=2)

# --- Main Dashboard ---
if selected_user_id is not None:
    u_gender = GENDER_MAP.get(user_meta_df.loc[selected_user_id, 'gender'], "N/A")
    u_age = AGE_MAP.get(user_meta_df.loc[selected_user_id, 'age'], "N/A")
    u_occ = OCCUPATION_MAP.get(user_meta_df.loc[selected_user_id, 'occupation'], "N/A")
    u_count = user_watch_counts.get(selected_user_id, 0)

    # Active User Profile Summary
    st.info(f"**Selected Persona:** {u_gender} | **Age:** {u_age} | **Occupation:** {u_occ} | **History:** {u_count} Rated Movies")

    # SECTION 1: WATCH HISTORY CARDS
    st.markdown("---")
    st.subheader("📜 Viewer Favorite History")
    history_items = get_user_watched_history(selected_user_id, train_ratings, item_genre_matrix, idx_to_title, top_n=6)
    
    # Render History as a 3-column Grid
    hist_cols = st.columns(3)
    for idx, item in enumerate(history_items):
        col = hist_cols[idx % 3]
        genre_badges = "".join([f'<span class="genre-tag">{g}</span>' for g in item['genres']])
        col.markdown(f"""
            <div class="movie-card">
                <div class="movie-title">🎬 {item['title']}</div>
                <div class="card-footer">
                    <div>{genre_badges}</div>
                </div>
            </div>
        """, unsafe_allow_html=True)

    # SECTION 2: RECOMMENDATIONS GRID CARDS
    st.markdown("---")
    st.subheader(f"🤖 Top-{top_k} Neural AI Recommendations")
    
    with st.spinner("Calculating neural matrix predictions..."):
        recs = recommend_for_user(
            user_id=selected_user_id,
            model=model,
            train_ratings=train_ratings,
            user_meta_df=user_meta_df,
            item_genre_matrix=item_genre_matrix,
            idx_to_title=idx_to_title,
            num_items=num_items,
            top_k=top_k,
            device=device
        )

    # Render Recommendations as a 2-column Visual Grid
    rec_cols = st.columns(2)
    for idx, rec in enumerate(recs):
        col = rec_cols[idx % 2]
        genre_badges = "".join([f'<span class="genre-tag">{g}</span>' for g in rec['genres']])
        
        col.markdown(f"""
            <div class="movie-card">
                <div style="display: flex; justify-content: space-between; align-items: center;">
                    <span class="rank-badge">RANK #{rec['rank']}</span>
                    <span class="match-score">🔥 {rec['match']}% Match</span>
                </div>
                <div class="movie-title">{rec['title']}</div>
                <div class="card-footer">
                    <div>{genre_badges}</div>
                </div>
            </div>
        """, unsafe_allow_html=True)