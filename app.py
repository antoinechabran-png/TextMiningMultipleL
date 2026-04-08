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
    cleaned = []
    for w in words:
        lemma = lemmatizer.lemmatize(w)
        lemma = fragrance_merges.get(lemma, lemma)
        if (lemma not in base_stops and lemma not in custom_stops_set and len(lemma) > 2):
            cleaned.append(lemma)
    return " ".join(cleaned)

# --- Analysis Functions ---
def get_sentiment_words(text_series):
    words = " ".join(text_series).split()
    if not words: return [], []
    unique_words = list(set(words))
    scored = [(w, TextBlob(w).sentiment.polarity) for w in unique_words]
    pos = sorted([x for x in scored if x[1] > 0.1], key=lambda x: x[1], reverse=True)[:10]
    neg = sorted([x for x in scored if x[1] < -0.1], key=lambda x: x[1])[:10]
    return pos, neg

def generate_word_cloud(text_series, palette, shape, allow_collocations):
    combined_text = " ".join(text_series).strip()
    if not combined_text:
        fig, ax = plt.subplots(); ax.text(0.5, 0.5, "No text available", ha='center'); ax.axis("off")
        return fig
    mask = None
    if shape == "Round":
        img = Image.new("L", (800, 800), 255)
        draw = ImageDraw.Draw(img); draw.ellipse((20,20,780,780), fill=0); mask = np.array(img)
    wc = WordCloud(background_color="white", colormap=palette, mask=mask, width=800, height=500, collocations=allow_collocations)
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

def run_fca(df, p_col, fmin, use_tfidf, ngrams):
    grouped = df.groupby(p_col)['cleaned'].apply(lambda x: " ".join(x))
    if len(grouped) < 3: return None, "Need 3+ products for Factorial Mapping."
    VecClass = TfidfVectorizer if use_tfidf else CountVectorizer
    vec = VecClass(min_df=min(fmin, len(grouped)), ngram_range=ngrams) 
    X = vec.fit_transform(grouped).toarray()
    words, products = vec.get_feature_names_out(), grouped.index.tolist()
    X_centered = X - np.mean(X, axis=0)
    svd = TruncatedSVD(n_components=2, random_state=42)
    row_coords = svd.fit_transform(X_centered)
    col_coords = svd.components_.T * (np.std(row_coords) / (np.std(svd.components_.T) + 1e-9))
    return (row_coords, col_coords, products, words, svd.explained_variance_ratio_), None

# --- UI Setup ---
with st.sidebar:
    st.header("⚙️ Settings")
    uploaded_file = st.file_uploader("Upload Excel", type=["xlsx"])
    
    if uploaded_file:
        df_raw = pd.read_excel(uploaded_file)
        
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
        st.subheader("🧪 Text Mining Logic")
        ngram_choice = st.radio("Active Grams:", ["1-Gram (Single Words)", "1 & 2-Grams (Pairs)", "1, 2 & 3-Grams (Phrases)"])
        ngram_map = {
            "1-Gram (Single Words)": (1, 1),
            "1 & 2-Grams (Pairs)": (1, 2),
            "1, 2 & 3-Grams (Phrases)": (1, 3)
        }
        selected_ngram_range = ngram_map[ngram_choice]
        allow_collocations = True if selected_ngram_range[1] > 1 else False

        st.divider()
        dataset_lang = st.selectbox("Language:", list(MULTILINGUAL_STOPWORDS.keys()))
        if 'custom_stop_list' not in st.session_state:
            st.session_state.custom_stop_list = MULTILINGUAL_STOPWORDS[dataset_lang]

        fmin_global = st.slider("Min Word Frequency", 1, 50, 5)
        use_tfidf = st.toggle("Use TF-IDF Weighting", value=True)
        shape_opt = st.radio("Cloud Shape", ["Rectangle", "Round"])
        palette_opt = st.selectbox("Palette", ["copper", "GnBu", "RdPu", "viridis"])

tab1, tab2, tab3, tab4, tab5 = st.tabs(["📊 Single Product", "⚔️ Comparison", "🌐 Factorial Map", "🔍 Topic Lab", "🚫 Exclusions"])

if uploaded_file:
    p_col = st.sidebar.selectbox("Product ID Column", df_raw.columns)
    v_col = st.sidebar.selectbox("Verbatim Column", df_raw.columns)

    if st.sidebar.button("🚀 Run Analysis on Sub-Target"):
        df_filtered = df_raw.loc[target_indices].dropna(subset=[v_col])
        df_filtered['cleaned'] = df_filtered[v_col].apply(lambda x: clean_text(x, st.session_state.custom_stop_list, dataset_lang))
        st.session_state['processed_df'] = df_filtered
        st.session_state['filter_info'] = filter_label

    if 'processed_df' in st.session_state:
        df = st.session_state['processed_df']
        p_list = sorted(df[p_col].dropna().astype(str).unique())
        
        st.caption(f"📍 **Currently Analyzing:** {st.session_state.get('filter_info', 'Total Sample')} (N={len(df)})")

        with tab1:
            target_p = st.selectbox("Fragrance Focus", p_list)
            product_data = df[df[p_col].astype(str) == target_p]
            p_sub_cleaned = product_data['cleaned']
            
            sent_val = product_data[v_col].apply(lambda x: TextBlob(str(x)).sentiment.polarity).mean()
            st.metric(f"Target Mood: {target_p}", f"{'Positive' if sent_val > 0 else 'Negative'}", f"{round(sent_val*100, 1)}%")
            st.progress((sent_val + 1) / 2)
            
            c1, c2 = st.columns(2)
            with c1: st.pyplot(generate_word_cloud(p_sub_cleaned, palette_opt, shape_opt, allow_collocations))
            with c2: 
                tree_fig = generate_word_tree(p_sub_cleaned, fmin_global, palette_opt)
                if tree_fig: st.pyplot(tree_fig)
                else: st.warning("Not enough patterns in this sub-target.")

            pos_words, neg_words = get_sentiment_words(p_sub_cleaned)
            l, r = st.columns(2)
            with l:
                st.success("✨ **Sub-Target Positive Descriptors**")
                for w, s in pos_words: st.write(f"- {w}")
            with r:
                st.error("⚠️ **Sub-Target Negative Descriptors**")
                for w, s in neg_words: st.write(f"- {w}")

        with tab2:
            st.subheader("⚔️ Scent Comparison (Sub-Target Only)")
            comp_cols = st.columns(2)
            p_a = comp_cols[0].selectbox("Fragrance A", p_list, index=0)
            p_b = comp_cols[1].selectbox("Fragrance B", p_list, index=min(1, len(p_list)-1))
            
            d_a = df[df[p_col].astype(str) == p_a]['cleaned']
            d_b = df[df[p_col].astype(str) == p_b]['cleaned']
            
            if not d_a.empty and not d_b.empty:
                sim = float(cosine_similarity(TfidfVectorizer(ngram_range=selected_ngram_range).fit_transform([" ".join(d_a), " ".join(d_b)]))[0][1])
                st.metric("Sub-Target Olfactive Similarity", f"{round(sim*100, 1)}%")
                comp_cols[0].pyplot(generate_word_cloud(d_a, palette_opt, shape_opt, allow_collocations))
                comp_cols[1].pyplot(generate_word_cloud(d_b, palette_opt, shape_opt, allow_collocations))

        with tab3:
            st.subheader("🌐 Factorial Mapping (Sub-Target Only)")
            res, err = run_fca(df, p_col, fmin_global, use_tfidf, selected_ngram_range)
            if not err:
                r_c, c_c, prods, wrds, _ = res
                fig, ax = plt.subplots(figsize=(10, 6))
                ax.scatter(r_c[:,0], r_c[:,1], c='blue', s=100)
                for i, txt in enumerate(prods): ax.text(r_c[i,0], r_c[i,1], txt, fontsize=12)
                ax.scatter(c_c[:,0], c_c[:,1], c='red', marker='x', alpha=0.2)
                for i, txt in enumerate(wrds):
                    if np.linalg.norm(c_c[i]) > np.percentile([np.linalg.norm(c) for c in c_c], 80):
                        ax.text(c_c[i,0], c_c[i,1], txt, color='darkred', fontsize=8)
                st.pyplot(fig)
            else: st.error(err)

        with tab4:
            st.subheader("🔍 Topic Lab (Sub-Target Only)")
            num_t = st.slider("Themes", 2, 8, 3)
            if st.button("Generate Topics"):
                vec = TfidfVectorizer(max_features=500, ngram_range=selected_ngram_range)
                mtx = vec.fit_transform(df['cleaned'])
                nmf = NMF(n_components=num_t, random_state=42, init='nndsvd').fit(mtx)
                
                doc_topic = nmf.transform(mtx)
                fn = vec.get_feature_names_out()
                cols = st.columns(num_t)
                
                for i, topic in enumerate(nmf.components_):
                    with cols[i % num_t]:
                        top_words = [fn[j] for j in topic.argsort()[-7:]]
                        st.info(f"**Theme {i+1}**\n\n" + ", ".join(top_words))
                        
                        closest_idx = doc_topic[:, i].argmax()
                        furthest_idx = doc_topic[:, i].argmin()
                        
                        lead_prod = df.iloc[closest_idx][p_col]
                        dist_prod = df.iloc[furthest_idx][p_col]
                        
                        st.success(f"✅ **Closest:** {lead_prod}")
                        st.error(f"❌ **Furthest:** {dist_prod}")

with tab5:
    st.subheader("🚫 Exclusions")
    stops = st.session_state.get('custom_stop_list', [])
    txt = st.text_area("Edit exclusions (comma separated)", value=", ".join(stops))
    if st.button("Apply Exclusions"):
        st.session_state.custom_stop_list = [x.strip().lower() for x in txt.split(",") if x.strip()]
        st.rerun()
