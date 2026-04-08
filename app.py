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

# --- Font Injection ---
def apply_custom_font(font_name):
    font_css = f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;700&family=Playfair+Display:wght@400;700&display=swap');
    html, body, [class*="css"], .stText, .stMarkdown {{ font-family: '{font_name}', sans-serif; }}
    </style>
    """
    st.markdown(font_css, unsafe_allow_html=True)

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

def generate_word_cloud(text_series, palette, shape):
    combined_text = " ".join(text_series).strip()
    if not combined_text:
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, "No descriptive words found for this sub-target.", ha='center', va='center')
        ax.axis("off")
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
    vec = VecClass(min_df=min_freq_adapted(fmin, len(grouped))) 
    X = vec.fit_transform(grouped).toarray()
    words, products = vec.get_feature_names_out(), grouped.index.tolist()
    X_centered = X - np.mean(X, axis=0)
    svd = TruncatedSVD(n_components=2, random_state=42)
    row_coords = svd.fit_transform(X_centered)
    col_coords = svd.components_.T * (np.std(row_coords) / (np.std(svd.components_.T) + 1e-9))
    return (row_coords, col_coords, products, words, svd.explained_variance_ratio_), None

def min_freq_adapted(slider_val, sample_size):
    # Helps prevent crashes if sub-target sample size is smaller than the slider frequency
    return min(slider_val, max(1, sample_size))

# --- UI Setup ---
with st.sidebar:
    st.header("⚙️ Global Settings")
    uploaded_file = st.file_uploader("Upload Excel", type=["xlsx"])
    
    if uploaded_file:
        df_raw = pd.read_excel(uploaded_file)
        
        st.subheader("🎯 Sub-Target Filter")
        filter_col = st.selectbox("Identify Sub-Target via Column:", ["No Filter"] + list(df_raw.columns))
        
        target_indices = df_raw.index
        if filter_col != "No Filter":
            options = sorted(df_raw[filter_col].dropna().unique())
            selected_codes = st.multiselect("Select Codes to Include:", options)
            if selected_codes:
                target_indices = df_raw[df_raw[filter_col].isin(selected_codes)].index
                st.success(f"Targeting {len(target_indices)} respondents")

        st.divider()
        dataset_lang = st.selectbox("Dataset Language:", list(MULTILINGUAL_STOPWORDS.keys()))
        
        if 'current_lang' not in st.session_state or st.session_state.current_lang != dataset_lang:
            st.session_state.current_lang = dataset_lang
            st.session_state.custom_stop_list = MULTILINGUAL_STOPWORDS[dataset_lang]

        st.subheader("🎨 Styling & Math")
        selected_font = st.selectbox("App Font:", ["Inter", "Helvetica Neue", "Playfair Display", "Clash Display"])
        apply_custom_font(selected_font)

        use_tfidf = st.toggle("Use TF-IDF Weighting", value=True)
        fmin_global = st.slider("Min Word Frequency", 1, 50, 5)
        shape_opt = st.radio("Cloud Shape", ["Rectangle", "Round"])
        palette_opt = st.selectbox("Color Palette", ["copper", "GnBu", "RdPu", "viridis"])

tab1, tab2, tab3, tab4, tab5 = st.tabs(["📊 Single Product", "⚔️ Comparison", "🌐 Factorial Map", "🔍 Topic Lab", "🚫 Exclusions"])

if uploaded_file:
    p_col = st.sidebar.selectbox("Product ID Column", df_raw.columns)
    v_col = st.sidebar.selectbox("Verbatim Column", df_raw.columns)

    if st.sidebar.button("🚀 Run Analysis on Target"):
        # Apply the sub-target filtering here
        df_filtered = df_raw.loc[target_indices].dropna(subset=[v_col])
        df_filtered['cleaned'] = df_filtered[v_col].apply(lambda x: clean_text(x, st.session_state.custom_stop_list, dataset_lang))
        st.session_state['processed_df'] = df_filtered

    if 'processed_df' in st.session_state:
        df = st.session_state['processed_df']
        p_list = sorted(df[p_col].dropna().astype(str).unique())

        if df.empty:
            st.warning("The selected sub-target results in an empty dataset. Please check your filters.")
        else:
            with tab1:
                target = st.selectbox("Fragrance Focus", p_list, key="single_focus")
                p_sub = df[df[p_col].astype(str) == target]['cleaned']
                
                sent_val = df[df[p_col].astype(str) == target][v_col].apply(lambda x: TextBlob(str(x)).sentiment.polarity).mean()
                st.metric(f"Target Mood for {target}", f"{'Positive' if sent_val > 0 else 'Negative'}", f"{round(sent_val*100, 1)}%")
                st.progress((sent_val + 1) / 2)
                st.divider()

                c1, c2 = st.columns(2)
                with c1: 
                    st.write("**Word Cloud**")
                    st.pyplot(generate_word_cloud(p_sub, palette_opt, shape_opt))
                with c2: 
                    st.write("**Word Tree (Scent Accords)**")
                    tree_fig = generate_word_tree(p_sub, fmin_global, palette_opt)
                    if tree_fig: st.pyplot(tree_fig)
                    else: st.warning("Not enough repeated patterns for this sub-target.")

                st.divider()
                pos_words, neg_words = get_sentiment_words(p_sub)
                l_col, r_col = st.columns(2)
                with l_col:
                    st.success("✨ **Target Positive Descriptors**")
                    if pos_words: 
                        for w, s in pos_words: st.write(f"- {w}")
                    else: st.caption("No strong positive patterns.")
                with r_col:
                    st.error("⚠️ **Target Negative Descriptors**")
                    if neg_words:
                        for w, s in neg_words: st.write(f"- {w}")
                    else: st.caption("No strong negative patterns.")

            with tab2:
                st.subheader("⚔️ Scent Comparison (Sub-Target Only)")
                comp_cols = st.columns(2)
                prod_a = comp_cols[0].selectbox("Fragrance A", p_list, index=0)
                prod_b = comp_cols[1].selectbox("Fragrance B", p_list, index=min(1, len(p_list)-1))
                
                data_a = df[df[p_col].astype(str) == prod_a]['cleaned']
                data_b = df[df[p_col].astype(str) == prod_b]['cleaned']
                
                if not data_a.empty and not data_b.empty:
                    v_comp = TfidfVectorizer().fit_transform([" ".join(data_a), " ".join(data_b)])
                    sim_score = float(cosine_similarity(v_comp[0], v_comp[1])[0][0])
                    st.metric("Olfactive Similarity", f"{round(sim_score*100, 1)}%")
                    st.progress(sim_score)
                    st.divider()
                    comp_cols[0].pyplot(generate_word_cloud(data_a, palette_opt, shape_opt))
                    comp_cols[1].pyplot(generate_word_cloud(data_b, palette_opt, shape_opt))

            with tab3:
                st.subheader("🌐 Factorial Mapping (Sub-Target Only)")
                res, err = run_fca(df, p_col, fmin_global, use_tfidf)
                if not err:
                    r_c, c_c, prods, wrds, var = res
                    fig, ax = plt.subplots(figsize=(12, 8))
                    ax.scatter(r_c[:,0], r_c[:,1], c='blue', s=150, alpha=0.7)
                    for i, txt in enumerate(prods): ax.text(r_c[i,0]+0.02, r_c[i,1]+0.02, txt, fontweight='bold')
                    ax.scatter(c_c[:,0], c_c[:,1], c='red', marker='x', alpha=0.3)
                    for i, txt in enumerate(wrds):
                        if np.linalg.norm(c_c[i]) > np.percentile([np.linalg.norm(c) for c in c_c], 70):
                            ax.text(c_c[i,0], c_c[i,1], txt, color='darkred', fontsize=9)
                    st.pyplot(fig)
                else: st.error(err)

            with tab4:
                st.subheader("🔍 Topic Lab")
                num_t = st.slider("Number of Themes", 2, 8, 4)
                if st.button("Generate Topic Models"):
                    vec_t = TfidfVectorizer(max_features=1000)
                    mtx_t = vec_t.fit_transform(df['cleaned'])
                    nmf = NMF(n_components=num_t, random_state=42, init='nndsvd').fit(mtx_t)
                    feature_names = vec_t.get_feature_names_out()
                    doc_topic = nmf.transform(mtx_t)
                    
                    t_cols = st.columns(min(num_t, 3))
                    for i, topic in enumerate(nmf.components_):
                        with t_cols[i % 3]:
                            top_words = [feature_names[j] for j in topic.argsort()[-10:]]
                            st.info(f"**Theme {i+1}**\n\n" + ", ".join(top_words))
                            lead_idx = doc_topic[:, i].argmax()
                            lead_prod = df.iloc[lead_idx][p_col]
                            st.caption(f"📍 **Lead Product:** {lead_prod}")

with tab5:
    st.subheader("🚫 Exclusions")
    default_stops = st.session_state.get('custom_stop_list', MULTILINGUAL_STOPWORDS["English"])
    txt = st.text_area("Edit exclusions", value=", ".join(default_stops), height=300)
    if st.button("Update Exclusions"):
        st.session_state.custom_stop_list = [x.strip().lower() for x in txt.split(",") if x.strip()]
        st.rerun()
