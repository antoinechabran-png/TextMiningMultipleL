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

def clean_text(text, custom_stops, lang_choice, gram_rules):
    if not text or pd.isna(text): return ""
    lang_map = {"English": "english", "French": "french", "German": "german", "Spanish": "spanish", "Portuguese": "portuguese", "Italian": "italian", "Indonesian": "indonesian"}
    try:
        base_stops = set(nltk.corpus.stopwords.words(lang_map.get(lang_choice, "english")))
    except:
        base_stops = set()

    custom_stops_set = set([str(x).strip().lower() for x in custom_stops])
    
    # Identify words that are part of any gram rule to avoid filtering them out too early
    gram_influencers = set()
    for key in ['prefix_2g', 'suffix_2g', 'prefix_3g', 'spec_2g', 'spec_3g']:
        for phrase in gram_rules[key]:
            for word in phrase.split():
                gram_influencers.add(word.lower())

    fragrance_merges = {"freshness": "fresh", "freshly": "fresh", "fruity": "fruit", "smelling": "smell", "scented": "scent", "floral": "flower", "flowers": "flower", "cleanliness": "clean", "cleaning": "clean"}

    words = re.findall(r'\b[a-zà-ÿ]{2,}\b', str(text).lower())
    
    tokens = []
    for w in words:
        lemma = lemmatizer.lemmatize(w)
        lemma = fragrance_merges.get(lemma, lemma)
        
        if lemma in custom_stops_set:
            if lemma not in gram_influencers:
                continue
            tokens.append(lemma)
        elif lemma not in base_stops or lemma in gram_influencers:
            tokens.append(lemma)
    
    if not tokens: return ""

    processed_tokens = []
    i = 0
    while i < len(tokens):
        match_found = False
        # Try Trigrams
        if i < len(tokens) - 2:
            trigram_raw = f"{tokens[i]} {tokens[i+1]} {tokens[i+2]}"
            prefix_2g_phrase = f"{tokens[i]} {tokens[i+1]}"
            if trigram_raw in gram_rules['spec_3g'] or prefix_2g_phrase in gram_rules['prefix_3g']:
                processed_tokens.append(f"{tokens[i]}_{tokens[i+1]}_{tokens[i+2]}")
                i += 3
                match_found = True
        
        # Try Bigrams
        if not match_found and i < len(tokens) - 1:
            bigram_raw = f"{tokens[i]} {tokens[i+1]}"
            if (bigram_raw in gram_rules['spec_2g'] or 
                tokens[i] in gram_rules['prefix_2g'] or 
                tokens[i+1] in gram_rules['suffix_2g']):
                processed_tokens.append(f"{tokens[i]}_{tokens[i+1]}")
                i += 2
                match_found = True
        
        if not match_found:
            if tokens[i] not in custom_stops_set:
                processed_tokens.append(tokens[i])
            i += 1
            
    return " ".join(processed_tokens)

def get_sentiment_words(text_series):
    words = " ".join(text_series).split()
    if not words: return [], []
    unique_words = list(set(words))
    scored = []
    for w in unique_words:
        display_text = w.replace("_", " ")
        score = TextBlob(display_text).sentiment.polarity
        scored.append((w, score))
    pos = sorted([x for x in scored if x[1] > 0.1], key=lambda x: x[1], reverse=True)[:10]
    neg = sorted([x for x in scored if x[1] < -0.1], key=lambda x: x[1])[:10]
    return pos, neg

def get_gram_categories(text_series, negation_list, superlative_list):
    all_text = " ".join(text_series)
    words = all_text.split()
    neg_captured = []
    sup_captured = []
    
    # Sync lists with underscore logic
    neg_set = set([n.strip().lower().replace(" ", "_") for n in negation_list])
    sup_set = set([s.strip().lower().replace(" ", "_") for s in superlative_list])

    for w in set(words):
        # Exact match or check if it starts with a negation (e.g., 'not_fresh')
        if w in neg_set or any(w.startswith(n + "_") for n in neg_set if "_" not in n):
            neg_captured.append(w.replace("_", " "))
        elif w in sup_set or any(w.startswith(s + "_") for s in sup_set if "_" not in s):
            sup_captured.append(w.replace("_", " "))
                
    return sorted(list(set(neg_captured)))[:15], sorted(list(set(sup_captured)))[:15]

def generate_word_cloud(text_series, palette, shape):
    combined_text = " ".join(text_series).strip()
    if not combined_text:
        fig, ax = plt.subplots(); ax.text(0.5, 0.5, "No text available", ha='center'); ax.axis("off")
        return fig
    mask = None
    if shape == "Round":
        img = Image.new("L", (800, 800), 255)
        draw = ImageDraw.Draw(img); draw.ellipse((20,20,780,780), fill=0); mask = np.array(img)
    
    # regexp=r"\S+" forces WordCloud to treat underscored words as a single token
    wc = WordCloud(
        background_color="white", 
        colormap=palette, 
        mask=mask, 
        width=800, 
        height=500, 
        collocations=False, 
        regexp=r"\S+"
    )
    wc.generate(combined_text)
    fig, ax = plt.subplots(); ax.imshow(wc, interpolation='bilinear'); ax.axis("off")
    return fig

def generate_word_tree(text_series, min_freq, palette):
    valid = [t for t in text_series if len(t.split()) > 1]
    if not valid: return None
    try:
        vec = CountVectorizer(min_df=min_freq, token_pattern=r"(?u)\b\S+\b")
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
    vec = VecClass(min_df=min(fmin, len(grouped)), token_pattern=r"(?u)\b\S+\b") 
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
        try:
            xl = pd.ExcelFile(uploaded_file)
            sheet = st.selectbox("Select Sheet:", xl.sheet_names)
            df_raw = pd.read_excel(uploaded_file, sheet_name=sheet)
            
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
        except Exception as e:
            st.error(f"Error loading file: {e}")
            st.stop()

        st.divider()
        dataset_lang = st.selectbox("Language:", list(MULTILINGUAL_STOPWORDS.keys()))
        if 'custom_stop_list' not in st.session_state:
            st.session_state.custom_stop_list = MULTILINGUAL_STOPWORDS[dataset_lang]

        fmin_global = st.slider("Min Word Frequency", 1, 50, 5)
        use_tfidf = st.toggle("Use TF-IDF Weighting", value=True)
        shape_opt = st.radio("Cloud Shape", ["Rectangle", "Round"])
        palette_opt = st.selectbox("Palette", ["copper", "GnBu", "RdPu", "viridis"])

if 'gram_rules' not in st.session_state:
    st.session_state.gram_rules = {
        'prefix_2g': ["not", "too", "very", "real", "really", "enough", "less", "more", "little", "lot", "so", "just", "quite"],
        'suffix_2g': ["not", "too", "very", "real", "really", "enough", "less", "more", "little", "lot", "so", "quite"],
        'prefix_3g': ["not too", "not very", "not real", "not enough"],
        'spec_2g': ["lily valley", "funeral flower", "white flower", "old fashion", "old people", "old lady", "house cleaner", "not fresh", "not clean"],
        'spec_3g': ["not smell good", "smell very good", "not smell bad", "smell very bad"],
        'negation_list': ["not", "not too", "less", "little", "not very", "not at all", "not fresh", "not clean"],
        'superlative_list': ["really", "very", "enough", "quite", "many", "just", "more", "real", "so", "too"]
    }

tab1, tab2, tab3, tab4, tab6, tab5 = st.tabs(["📊 Single Product", "⚔️ Comparison", "🌐 Factorial Map", "🔍 Topic Lab", "🎯 Impact Lab", "🚫 Exclusions & Grams"])

if uploaded_file and 'df_raw' in locals():
    p_col = st.sidebar.selectbox("Product ID Column", df_raw.columns)
    v_col = st.sidebar.selectbox("Verbatim Column", df_raw.columns)
    s_col = st.sidebar.selectbox("Preference Score (Optional)", ["None"] + list(df_raw.columns))

    if st.sidebar.button("🚀 Run Analysis on Sub-Target"):
        df_filtered = df_raw.loc[target_indices].dropna(subset=[v_col])
        df_filtered['cleaned'] = df_filtered[v_col].apply(lambda x: clean_text(x, st.session_state.custom_stop_list, dataset_lang, st.session_state.gram_rules))
        st.session_state['processed_df'] = df_filtered
        st.session_state['filter_info'] = filter_label
        st.session_state['pref_col'] = s_col

    if 'processed_df' in st.session_state:
        df = st.session_state['processed_df']
        p_list = sorted(df[p_col].dropna().astype(str).unique())
        st.caption(f"📍 **Analyzing:** {st.session_state.get('filter_info', 'Total Sample')} (N={len(df)})")

        with tab1:
            target_p = st.selectbox("Fragrance Focus", p_list)
            product_data = df[df[p_col].astype(str) == target_p]
            p_sub_cleaned = product_data['cleaned']
            
            if not p_sub_cleaned.empty:
                full_text = " ".join(p_sub_cleaned)
                cv = CountVectorizer(token_pattern=r"(?u)\b\S+\b")
                cv_mtx = cv.fit_transform([full_text])
                counts = dict(zip(cv.get_feature_names_out(), cv_mtx.toarray()[0]))
                
                export_df = pd.DataFrame({
                    "Word": [w.replace("_", " ") for w in counts.keys()],
                    "Frequency": counts.values()
                }).sort_values(by="Frequency", ascending=False)
                
                st.download_button("📥 Download Stats", data=export_df.to_csv(index=False), file_name=f"{target_p}_stats.csv")

                c1, c2 = st.columns(2)
                with c1: st.pyplot(generate_word_cloud(p_sub_cleaned, palette_opt, shape_opt))
                with c2: 
                    tree_fig = generate_word_tree(p_sub_cleaned, fmin_global, palette_opt)
                    if tree_fig: st.pyplot(tree_fig)
                    else: st.warning("Not enough patterns for a tree.")

                neg_grams, sup_grams = get_gram_categories(p_sub_cleaned, st.session_state.gram_rules['negation_list'], st.session_state.gram_rules['superlative_list'])
                l2, r2 = st.columns(2)
                with l2:
                    st.warning("🚫 **Negation List**")
                    if neg_grams:
                        for g in neg_grams: st.write(f"- {g}")
                    else: st.write("None found.")
                with r2:
                    st.info("💎 **Superlative List**")
                    if sup_grams:
                        for g in sup_grams: st.write(f"- {g}")
                    else: st.write("None found.")

        with tab5:
            st.subheader("🚫 Exclusions & Gram Lab")
            col_left, col_right = st.columns(2)
            with col_left:
                stops = st.session_state.get('custom_stop_list', [])
                txt_stops = st.text_area("Stopwords", value=", ".join(stops))
                gn_list = st.text_input("Negation Categories", ", ".join(st.session_state.gram_rules['negation_list']))
            with col_right:
                gs_list = st.text_input("Superlative Categories", ", ".join(st.session_state.gram_rules['superlative_list']))
                a2 = st.text_input("Special Bigrams", ", ".join(st.session_state.gram_rules['spec_2g']))

            if st.button("💾 Save & Re-Process"):
                st.session_state.gram_rules['negation_list'] = [x.strip().lower() for x in gn_list.split(",") if x.strip()]
                st.session_state.gram_rules['superlative_list'] = [x.strip().lower() for x in gs_list.split(",") if x.strip()]
                st.session_state.gram_rules['spec_2g'] = [x.strip().lower() for x in a2.split(",") if x.strip()]
                st.session_state.custom_stop_list = [x.strip().lower() for x in txt_stops.split(",") if x.strip()]
                st.rerun()

        # Placeholders for other tabs to keep code valid
        with tab2: st.write("Use Run Analysis to refresh comparisons.")
        with tab3: st.write("Factorial Map requires analysis run.")
        with tab4: st.write("Topic Lab requires analysis run.")
        with tab6: st.write("Impact Lab requires analysis run.")
