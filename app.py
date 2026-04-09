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
from sklearn.linear_model import Ridge
import re
import numpy as np
from PIL import Image, ImageDraw
from textblob import TextBlob
import io

# Page Config
st.set_page_config(page_title="Fragrance Verbatim Lab Pro", layout="wide", page_icon="🧪")

# --- NLP Engine ---
@st.cache_resource
def setup_nltk():
    nltk.download('wordnet', quiet=True)
    nltk.download('omw-1.4', quiet=True)
    nltk.download('stopwords', quiet=True)
    return WordNetLemmatizer()

lemmatizer = setup_nltk()

def clean_text(text, custom_stops, lang_choice, gram_rules):
    if not text or pd.isna(text): return ""
    lang_map = {"English": "english", "French": "french", "German": "german", "Spanish": "spanish", "Portuguese": "portuguese", "Italian": "italian", "Indonesian": "indonesian"}
    try:
        base_stops = set(nltk.corpus.stopwords.words(lang_map.get(lang_choice, "english")))
    except:
        base_stops = set()
    
    custom_stops_set = set([str(x).strip().lower() for x in custom_stops])
    fragrance_merges = {"freshness": "fresh", "freshly": "fresh", "fruity": "fruit", "smelling": "smell", "scented": "scent", "floral": "flower", "flowers": "flower", "cleanliness": "clean", "cleaning": "clean"}
    
    words = re.findall(r'\b[a-zà-ÿ]{2,}\b', str(text).lower())
    tokens = []
    for w in words:
        lemma = lemmatizer.lemmatize(w)
        lemma = fragrance_merges.get(lemma, lemma)
        if lemma not in custom_stops_set and lemma not in base_stops:
            tokens.append(lemma)
    return " ".join(tokens)

# --- Analysis Functions ---
def run_impact_analysis(df, text_col, score_col):
    """Calculates Linear Regression coefficients for word impact on scores."""
    vec = CountVectorizer(min_df=3, binary=True)
    X = vec.fit_transform(df[text_col])
    y = df[score_col]
    
    model = Ridge(alpha=1.0)
    model.fit(X, y)
    
    impact_df = pd.DataFrame({
        'Word': vec.get_feature_names_out(),
        'Impact': model.coef_
    }).sort_values(by='Impact', ascending=False)
    
    return impact_df

def get_sentiment_words(text_series):
    words = " ".join(text_series).split()
    if not words: return [], []
    unique_words = list(set(words))
    scored = [(w, TextBlob(w.replace("_", " ")).sentiment.polarity) for w in unique_words]
    pos = sorted([x for x in scored if x[1] > 0.1], key=lambda x: x[1], reverse=True)[:10]
    neg = sorted([x for x in scored if x[1] < -0.1], key=lambda x: x[1])[:10]
    return pos, neg

def generate_word_cloud(text_series, palette, shape):
    combined_text = " ".join(text_series).strip()
    if not combined_text:
        fig, ax = plt.subplots(); ax.text(0.5, 0.5, "No text available", ha='center'); ax.axis("off")
        return fig
    mask = None
    if shape == "Round":
        img = Image.new("L", (800, 800), 255)
        draw = ImageDraw.Draw(img); draw.ellipse((20,20,780,780), fill=0); mask = np.array(img)
    wc = WordCloud(background_color="white", colormap=palette, mask=mask, width=800, height=500, collocations=False)
    wc.generate(combined_text)
    fig, ax = plt.subplots(); ax.imshow(wc, interpolation='bilinear'); ax.axis("off")
    return fig

def generate_word_tree(text_series, min_freq, palette):
    valid = [t for t in text_series if len(t.split()) > 1]
    if not valid: return None
    try:
        vec = CountVectorizer(min_df=min_freq)
        mtx = vec.fit_transform(valid); words = vec.get_feature_names_out()
        if len(words) < 2: return None
        adj = (mtx.T * mtx); adj.setdiag(0); G = nx.from_scipy_sparse_array(adj)
        G = nx.relabel_nodes(G, {i: w for i, w in enumerate(words)})
        T = nx.maximum_spanning_tree(G)
        fig, ax = plt.subplots(figsize=(8,6))
        pos = nx.spring_layout(T, k=1.5, seed=42); part = community_louvain.best_partition(T)
        nx.draw_networkx_nodes(T, pos, node_size=2000, node_color=list(part.values()), cmap=palette, alpha=0.8)
        nx.draw_networkx_labels(T, pos, font_size=8, font_weight='bold'); nx.draw_networkx_edges(T, pos, alpha=0.2)
        plt.axis('off'); return fig
    except: return None

def run_fca(df, p_col, fmin, use_tfidf):
    grouped = df.groupby(p_col)['cleaned'].apply(lambda x: " ".join(x))
    if len(grouped) < 3: return None, "Need 3+ products for Factorial Mapping."
    VecClass = TfidfVectorizer if use_tfidf else CountVectorizer
    vec = VecClass(min_df=min(fmin, len(grouped))) 
    X = vec.fit_transform(grouped).toarray()
    words, products = vec.get_feature_names_out(), grouped.index.tolist()
    X_centered = X - np.mean(X, axis=0)
    svd = TruncatedSVD(n_components=2, random_state=42)
    row_coords = svd.fit_transform(X_centered)
    col_coords = svd.components_.T * (np.std(row_coords) / (np.std(svd.components_.T) + 1e-9))
    return (row_coords, col_coords, products, words, svd.explained_variance_ratio_), None

# --- UI Setup ---
MULTILINGUAL_STOPWORDS = {"English": ["product", "smell"], "French": ["produit"], "German": ["produkt"], "Spanish": ["producto"], "Portuguese": ["producto"], "Italian": ["prodotto"], "Indonesian": ["produk"]}

with st.sidebar:
    st.header("⚙️ Settings")
    uploaded_file = st.file_uploader("Upload Excel", type=["xlsx"])
    
    if uploaded_file:
        try:
            # 1. Sheet Selector
            xl = pd.ExcelFile(uploaded_file)
            sheet_name = st.selectbox("Select Sheet:", xl.sheet_names)
            df_raw = pd.read_excel(uploaded_file, sheet_name=sheet_name)
            
            st.subheader("🎯 Sub-Target Filter")
            filter_col = st.selectbox("Filter Column:", ["No Filter"] + list(df_raw.columns))
            target_indices = df_raw.index
            filter_label = "Total Sample"
            if filter_col != "No Filter":
                options = sorted(df_raw[filter_col].dropna().unique())
                selected_codes = st.multiselect("Select Codes:", options)
                if selected_codes:
                    target_indices = df_raw[df_raw[filter_col].isin(selected_codes)].index
                    filter_label = f"{filter_col}: {', '.join(map(str, selected_codes))}"
            
            st.divider()
            # 2. Variable Mapping
            p_col = st.selectbox("Product ID Column", df_raw.columns)
            v_col = st.selectbox("Verbatim Column", df_raw.columns)
            s_col = st.selectbox("Preference Score (Optional)", ["None"] + list(df_raw.columns))
            
        except Exception as e:
            st.error(f"Error reading file: {e}")
            st.stop()

        st.divider()
        dataset_lang = st.selectbox("Language:", list(MULTILINGUAL_STOPWORDS.keys()))
        fmin_global = st.slider("Min Word Frequency", 1, 50, 5)
        use_tfidf = st.toggle("Use TF-IDF Weighting", value=True)
        shape_opt = st.radio("Cloud Shape", ["Rectangle", "Round"])
        palette_opt = st.selectbox("Palette", ["copper", "GnBu", "RdPu", "viridis"])

if 'gram_rules' not in st.session_state:
    st.session_state.gram_rules = {'prefix_2g': ["not"], 'suffix_2g': ["not"], 'prefix_3g': [], 'spec_2g': [], 'spec_3g': []}

tab1, tab2, tab3, tab4, tab_impact, tab5 = st.tabs(["📊 Single Product", "⚔️ Comparison", "🌐 Factorial Map", "🔍 Topic Lab", "🎯 Impact Lab", "🚫 Exclusions"])

if uploaded_file and 'df_raw' in locals():
    if st.sidebar.button("🚀 Run Analysis"):
        df_filtered = df_raw.loc[target_indices].dropna(subset=[v_col])
        df_filtered['cleaned'] = df_filtered[v_col].apply(lambda x: clean_text(x, st.session_state.get('custom_stop_list', []), dataset_lang, st.session_state.gram_rules))
        st.session_state['processed_df'] = df_filtered
        st.session_state['filter_info'] = filter_label
        st.session_state['score_col'] = s_col

    if 'processed_df' in st.session_state:
        df = st.session_state['processed_df']
        p_list = sorted(df[p_col].dropna().astype(str).unique())
        
        with tab1:
            target_p = st.selectbox("Fragrance Focus", p_list)
            product_data = df[df[p_col].astype(str) == target_p]
            st.pyplot(generate_word_cloud(product_data['cleaned'], palette_opt, shape_opt))

        with tab2:
            st.subheader("⚔️ Scent Comparison")
            # Comparison Logic... (Existing code works here)

        with tab3:
            st.subheader("🌐 Factorial Mapping")
            res, err = run_fca(df, p_col, fmin_global, use_tfidf)
            if not err:
                r_c, c_c, prods, wrds, _ = res
                fig, ax = plt.subplots(figsize=(10, 6))
                ax.scatter(r_c[:,0], r_c[:,1], c='blue', s=100)
                for i, txt in enumerate(prods): ax.text(r_c[i,0], r_c[i,1], txt)
                st.pyplot(fig)

        with tab4:
            st.subheader("🔍 Topic Lab")
            # Topic logic... (Existing code works here)

        with tab_impact:
            st.subheader("🎯 Preference Driver Analysis")
            score_col = st.session_state.get('score_col')
            if score_col == "None" or score_col not in df.columns:
                st.warning("Please select a numerical 'Preference Score' column in the sidebar to unlock this analysis.")
            else:
                try:
                    df_impact = df.dropna(subset=[score_col, 'cleaned'])
                    df_impact = df_impact[df_impact['cleaned'] != ""]
                    
                    impact_results = run_impact_analysis(df_impact, 'cleaned', score_col)
                    
                    c1, c2 = st.columns(2)
                    with c1:
                        st.success("📈 **Positive Drivers** (Increases Score)")
                        st.dataframe(impact_results.head(15), use_container_width=True)
                    with c2:
                        st.error("📉 **Negative Drivers** (Decreases Score)")
                        st.dataframe(impact_results.tail(15).sort_values(by='Impact'), use_container_width=True)
                    
                    fig, ax = plt.subplots(figsize=(10, 8))
                    top_bottom = pd.concat([impact_results.head(10), impact_results.tail(10)])
                    colors = ['green' if x > 0 else 'red' for x in top_bottom['Impact']]
                    ax.barh(top_bottom['Word'], top_bottom['Impact'], color=colors)
                    ax.set_title(f"Correlation with {score_col}")
                    st.pyplot(fig)
                except Exception as e:
                    st.error(f"Could not run impact analysis. Ensure your score column contains numbers. Error: {e}")

with tab5:
    st.subheader("🚫 Exclusions & Gram Lab")
    # Exclusion UI... (Existing code works here)
