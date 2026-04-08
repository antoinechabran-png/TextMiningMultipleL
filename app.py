import streamlit as st
import pandas as pd
import nltk
from nltk.stem import WordNetLemmatizer
import networkx as nx
from community import community_louvain
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import TruncatedSVD, NMF
import re
import numpy as np
from PIL import Image, ImageDraw
from textblob import TextBlob

# Page Config
st.set_page_config(page_title="Fragrance Verbatim Lab Pro", layout="wide", page_icon="🧪")

# --- Multilingual Exclusion Dictionary ---
MULTILINGUAL_STOPWORDS = {
    "English": ["product", "smell", "feel", "really", "just", "like", "little", "think", "lot", "make", "also", "bit", "quite", "something", "seem", "evoke", "find", "remind"],
    "French": ["produit", "odeur", "sent", "vraiment", "comme", "plus", "bien", "fait", "tout", "après", "assez", "évoque", "trouve", "rappelle", "petit", "beaucoup", "être", "avoir"],
    "German": ["produkt", "riecht", "geruch", "wirklich", "ganz", "viel", "mehr", "oder", "etwa", "lässt", "erinnert", "finde", "bisschen", "scheint", "etwas", "gut", "immer"],
    "Spanish": ["producto", "huele", "olor", "muy", "como", "mas", "pero", "todo", "este", "sentir", "parece", "evoca", "encuentro", "recuerda", "poco", "mucho", "bien"],
    "Portuguese": ["producto", "cheiro", "sinto", "muito", "como", "mais", "mas", "tudo", "este", "parece", "evoca", "acho", "lembra", "pouco", "muito", "bem"],
    "Italian": ["prodotto", "odore", "sento", "molto", "come", "più", "ma", "tutto", "questo", "sembra", "evoca", "trovo", "ricorda", "poco", "molto", "bene"],
    "Indonesian": ["produk", "bau", "wangi", "sangat", "seperti", "lebih", "tapi", "semua", "ini", "merasa", "tampak", "mengingatkan", "sedikit", "banyak", "bagus"]
}

# --- NLP Engine ---
@st.cache_resource
def setup_nltk():
    nltk.download('wordnet', quiet=True)
    nltk.download('omw-1.4', quiet=True)
    nltk.download('stopwords', quiet=True)
    return WordNetLemmatizer()

lemmatizer = setup_nltk()

def clean_text(text, custom_stops, lang_choice):
    if not text or pd.isna(text): return ""
    lang_map = {"English": "english", "French": "french", "German": "german", "Spanish": "spanish", "Portuguese": "portuguese", "Italian": "italian", "Indonesian": "indonesian"}
    try:
        base_stops = set(nltk.corpus.stopwords.words(lang_map.get(lang_choice, "english")))
    except:
        base_stops = set()
    custom_stops_set = set([str(x).strip().lower() for x in custom_stops])
    fragrance_merges = {"freshness": "fresh", "freshly": "fresh", "fruity": "fruit", "smelling": "smell", "scented": "scent", "floral": "flower", "flowers": "flower", "cleanliness": "clean", "cleaning": "clean"}
    words = re.findall(r'\b[a-zà-ÿ]{3,}\b', str(text).lower())
    cleaned = [fragrance_merges.get(lemmatizer.lemmatize(w), lemmatizer.lemmatize(w)) for w in words if w not in base_stops and w not in custom_stops_set and len(w) > 2]
    return " ".join(cleaned)

# --- Analysis Functions ---
def get_sentiment_words(text_series):
    words = " ".join(text_series).split()
    if not words: return [], []
    scored = [(w, TextBlob(w).sentiment.polarity) for w in set(words)]
    pos = sorted([x for x in scored if x[1] > 0.15], key=lambda x: x[1], reverse=True)[:10]
    neg = sorted([x for x in scored if x[1] < -0.15], key=lambda x: x[1])[:10]
    return pos, neg

def generate_word_cloud(text_series, palette, shape):
    combined_text = " ".join(text_series).strip()
    if not combined_text:
        fig, ax = plt.subplots(); ax.text(0.5, 0.5, "No Data", ha='center'); ax.axis("off"); return fig
    mask = None
    if shape == "Round":
        img = Image.new("L", (800, 800), 255)
        draw = ImageDraw.Draw(img); draw.ellipse((20,20,780,780), fill=0); mask = np.array(img)
    wc = WordCloud(background_color="white", colormap=palette, mask=mask, width=800, height=500, collocations=False).generate(combined_text)
    fig, ax = plt.subplots(); ax.imshow(wc, interpolation='bilinear'); ax.axis("off"); return fig

# --- UI Setup ---
with st.sidebar:
    st.header("⚙️ Global Settings")
    uploaded_file = st.file_uploader("Upload Excel", type=["xlsx"])
    
    if uploaded_file:
        df_raw = pd.read_excel(uploaded_file)
        st.subheader("🎯 Sub-Target Filter")
        filter_col = st.selectbox("Column:", ["No Filter"] + list(df_raw.columns))
        
        # Determine filtering
        if filter_col != "No Filter":
            opts = sorted(df_raw[filter_col].dropna().unique())
            selected_codes = st.multiselect("Select Codes:", opts)
            if selected_codes:
                st.session_state['active_indices'] = df_raw[df_raw[filter_col].isin(selected_codes)].index
                st.success(f"Filter active: {len(st.session_state['active_indices'])} rows")
            else:
                st.session_state['active_indices'] = df_raw.index
        else:
            st.session_state['active_indices'] = df_raw.index

        dataset_lang = st.selectbox("Language:", list(MULTILINGUAL_STOPWORDS.keys()))
        if 'custom_stop_list' not in st.session_state:
            st.session_state.custom_stop_list = MULTILINGUAL_STOPWORDS[dataset_lang]

        fmin_global = st.slider("Min Frequency", 1, 20, 3)
        palette_opt = st.selectbox("Palette", ["copper", "GnBu", "RdPu", "viridis"])
        shape_opt = st.radio("Shape", ["Rectangle", "Round"])

tab1, tab2, tab3, tab4, tab5 = st.tabs(["📊 Single Product", "⚔️ Comparison", "🌐 Factorial Map", "🔍 Topic Lab", "🚫 Exclusions"])

if uploaded_file:
    p_col = st.sidebar.selectbox("Product ID Column", df_raw.columns)
    v_col = st.sidebar.selectbox("Verbatim Column", df_raw.columns)

    if st.sidebar.button("🚀 Run Analysis"):
        # CRITICAL: Create the filtered working dataset here
        filtered_df = df_raw.loc[st.session_state['active_indices']].dropna(subset=[v_col])
        filtered_df['cleaned'] = filtered_df[v_col].apply(lambda x: clean_text(x, st.session_state.custom_stop_list, dataset_lang))
        st.session_state['processed_df'] = filtered_df

    if 'processed_df' in st.session_state:
        df = st.session_state['processed_df']
        p_list = sorted(df[p_col].dropna().astype(str).unique())

        with tab1:
            target_p = st.selectbox("Fragrance Focus", p_list)
            # Filter the already-filtered df for this specific fragrance
            p_data = df[df[p_col].astype(str) == target_p]
            
            # Recalculate Mood
            mood_score = p_data[v_col].apply(lambda x: TextBlob(str(x)).sentiment.polarity).mean()
            st.metric(f"Sub-Target Mood for {target_p}", f"{'Positive' if mood_score > 0 else 'Negative'}", f"{round(mood_score*100, 1)}%")
            st.progress((mood_score + 1) / 2)
            
            c1, c2 = st.columns(2)
            with c1: st.pyplot(generate_word_cloud(p_data['cleaned'], palette_opt, shape_opt))
            with c2:
                pos, neg = get_sentiment_words(p_data['cleaned'])
                st.success("✨ **Sub-Target Positive Descriptors**")
                if pos: 
                    for w, s in pos: st.write(f"- {w}")
                else: st.caption("No patterns found.")
                st.error("⚠️ **Sub-Target Negative Descriptors**")
                if neg:
                    for w, s in neg: st.write(f"- {w}")
                else: st.caption("No patterns found.")

        with tab4:
            st.subheader("🔍 Topic Lab (Sub-Target Only)")
            num_t = st.slider("Number of Themes", 2, 8, 4)
            if st.button("Generate Topic Models"):
                vec = TfidfVectorizer(max_features=1000)
                mtx = vec.fit_transform(df['cleaned'])
                nmf = NMF(n_components=num_t, random_state=42, init='nndsvd').fit(mtx)
                
                # Get topic-to-document scores to find lead fragrances
                doc_topic_matrix = nmf.transform(mtx)
                feature_names = vec.get_feature_names_out()
                
                t_cols = st.columns(min(num_t, 3))
                for i, topic in enumerate(nmf.components_):
                    with t_cols[i % 3]:
                        # Get top words for theme
                        top_words = [feature_names[j] for j in topic.argsort()[-10:]]
                        st.info(f"**Theme {i+1}**\n\n" + ", ".join(top_words))
                        
                        # Find the Lead Fragrance (highest score for this topic)
                        lead_idx = doc_topic_matrix[:, i].argmax()
                        lead_fragrance = df.iloc[lead_idx][p_col]
                        st.success(f"📍 **Lead Fragrance:** {lead_fragrance}")

with tab5:
    st.subheader("🚫 Exclusions")
    stops = st.session_state.get('custom_stop_list', [])
    txt = st.text_area("Edit exclusions", value=", ".join(stops))
    if st.button("Update Exclusions"):
        st.session_state.custom_stop_list = [x.strip().lower() for x in txt.split(",") if x.strip()]
        st.rerun()
