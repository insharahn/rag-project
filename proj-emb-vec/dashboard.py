# dashboard.py
# Run: streamlit run dashboard.py

import json
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path

st.set_page_config(
    page_title="Retrieval Benchmark",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .block-container { padding-top: 2rem; padding-bottom: 2rem; max-width: 1200px; }
    h1 { font-weight: 600; font-size: 1.8rem; }
    h2 { font-weight: 500; border-bottom: 1px solid #e9ecef; padding-bottom: 0.4rem; margin-top: 1.5rem; }
    h3 { font-weight: 500; }
    .metric-card {
        background: #f8f9fa;
        border: 1px solid #dee2e6;
        border-radius: 6px;
        padding: 1rem 1.25rem;
        height: 100%;
    }
    .metric-card .label { font-size: 0.75rem; color: #6c757d; text-transform: uppercase; letter-spacing: 0.06em; margin-bottom: 0.25rem; }
    .metric-card .value { font-size: 1.6rem; font-weight: 600; color: #212529; line-height: 1.2; }
    .metric-card .sub { font-size: 0.8rem; color: #6c757d; margin-top: 0.25rem; }
    .metric-card-green {
        background: #f0faf3;
        border: 1px solid #b7dfbf;
        border-radius: 6px;
        padding: 1rem 1.25rem;
        height: 100%;
    }
    .metric-card-green .label { font-size: 0.75rem; color: #4a7c59; text-transform: uppercase; letter-spacing: 0.06em; margin-bottom: 0.25rem; }
    .metric-card-green .value { font-size: 1.6rem; font-weight: 600; color: #2d6a4f; line-height: 1.2; }
    .metric-card-green .sub { font-size: 0.8rem; color: #4a7c59; margin-top: 0.25rem; }
    .metric-card-blue {
        background: #f0f6fc;
        border: 1px solid #b3cde8;
        border-radius: 6px;
        padding: 1rem 1.25rem;
        height: 100%;
    }
    .metric-card-blue .label { font-size: 0.75rem; color: #3a6186; text-transform: uppercase; letter-spacing: 0.06em; margin-bottom: 0.25rem; }
    .metric-card-blue .value { font-size: 1.6rem; font-weight: 600; color: #2c5282; line-height: 1.2; }
    .metric-card-blue .sub { font-size: 0.8rem; color: #3a6186; margin-top: 0.25rem; }
    .callout {
        background: #f8f9fa;
        border-left: 3px solid #adb5bd;
        padding: 0.75rem 1rem;
        border-radius: 0 4px 4px 0;
        font-size: 0.875rem;
        color: #495057;
        margin: 0.75rem 0 1rem 0;
        line-height: 1.5;
    }
    .callout-warn {
        background: #fff9db;
        border-left: 3px solid #f59f00;
        padding: 0.75rem 1rem;
        border-radius: 0 4px 4px 0;
        font-size: 0.875rem;
        color: #495057;
        margin: 0.75rem 0 1rem 0;
    }
    [data-testid="stSidebar"] { background: #f8f9fa; border-right: 1px solid #dee2e6; }
    [data-testid="stSidebar"] .stRadio label { font-size: 0.9rem; }
</style>
""", unsafe_allow_html=True)

ROOT = Path(__file__).resolve().parent

STRATEGY_PATHS = {
    "semantic":  ROOT / "results"           / "metrics.json",
    "fixed":     ROOT / "results_fixed"     / "metrics.json",
    "recursive": ROOT / "results_recursive" / "metrics.json",
}

STRATEGY_LABELS = {
    "semantic":  "Semantic",
    "fixed":     "Fixed-size",
    "recursive": "Recursive",
}

STRATEGY_CHUNKS = {
    "semantic":  10456,
    "fixed":     6091,
    "recursive": 7325,
}

# ms/chunk per (strategy, model) from actual benchmark runs
SPEED_DATA = {
    ("semantic",  "bge-large"):             {"ms_per_chunk": 101, "total_s": 1055.3, "type": "English-only"},
    ("semantic",  "bge-m3"):                {"ms_per_chunk": 93,  "total_s": 975.3,  "type": "Multilingual"},
    ("semantic",  "e5-large"):              {"ms_per_chunk": 103, "total_s": 1074.7, "type": "English-only"},
    ("semantic",  "multilingual-e5-large"): {"ms_per_chunk": 84,  "total_s": 879.6,  "type": "Multilingual"},
    ("semantic",  "instructor-xl"):         {"ms_per_chunk": 364, "total_s": 3807.1, "type": "Instruction-based"},
    ("fixed",     "bge-large"):             {"ms_per_chunk": 107, "total_s": 649.8,  "type": "English-only"},
    ("fixed",     "bge-m3"):                {"ms_per_chunk": 129, "total_s": 783.7,  "type": "Multilingual"},
    ("fixed",     "e5-large"):              {"ms_per_chunk": 102, "total_s": 620.8,  "type": "English-only"},
    ("fixed",     "multilingual-e5-large"): {"ms_per_chunk": 95,  "total_s": 580.4,  "type": "Multilingual"},
    ("fixed",     "instructor-xl"):         {"ms_per_chunk": 536, "total_s": 3267.0, "type": "Instruction-based"},
    ("recursive", "bge-large"):             {"ms_per_chunk": 103, "total_s": 751.8,  "type": "English-only"},
    ("recursive", "bge-m3"):                {"ms_per_chunk": 107, "total_s": 786.7,  "type": "Multilingual"},
    ("recursive", "e5-large"):              {"ms_per_chunk": 103, "total_s": 751.7,  "type": "English-only"},
    ("recursive", "multilingual-e5-large"): {"ms_per_chunk": 94,  "total_s": 687.3,  "type": "Multilingual"},
    ("recursive", "instructor-xl"):         {"ms_per_chunk": 502, "total_s": 3680.5, "type": "Instruction-based"},
}

# Pastel palette
PASTEL = {
    "blue":   "#B1E6F3",
    "green":  "#D0F0C0",
    "orange": "#FFBA8E",
    "purple": "#D4C4EC",
    "red":    "#D98585",
    "yellow": "#ffee8c",
    "pink": "#FFC2D1",
}

STRATEGY_COLORS = {
    "semantic":  PASTEL["blue"],
    "fixed":     PASTEL["pink"],
    "recursive": PASTEL["green"],
}

STRATEGY_COLOR_LIST = [PASTEL["blue"], PASTEL["orange"], PASTEL["green"]]

DB_COLORS = {
    "faiss":  PASTEL["yellow"],
    "chroma": PASTEL["pink"],
    "milvus": PASTEL["orange"],
}

LANG_COLORS = {
    "EN": PASTEL["blue"],
    "KO": PASTEL["green"],
    "UR": PASTEL["red"],
}

TYPE_COLORS = {
    "Multilingual":       PASTEL["blue"],
    "English-only":       PASTEL["orange"],
    "Instruction-based":  PASTEL["red"],
}


@st.cache_data
def load_all_metrics():
    out = {}
    for strategy, path in STRATEGY_PATHS.items():
        if path.exists():
            out[strategy] = json.loads(path.read_text(encoding="utf-8"))
    return out


all_data = load_all_metrics()
available = list(all_data.keys())

if not all_data:
    st.error("No metrics.json files found. Run scripts/report_metrics.py first.")
    st.stop()


def note(text, warn=False):
    cls = "callout-warn" if warn else "callout"
    st.markdown(f'<div class="{cls}">{text}</div>', unsafe_allow_html=True)


def clean_fig(fig):
    fig.update_layout(
        plot_bgcolor="white",
        paper_bgcolor="white",
        font_family="sans-serif",
        font_size=12,
        margin=dict(t=45, b=30, l=10, r=10),
        legend=dict(bgcolor="rgba(0,0,0,0)", borderwidth=0),
    )
    fig.update_xaxes(showgrid=False, linecolor="#dee2e6", linewidth=1, tickfont_size=11)
    fig.update_yaxes(gridcolor="#f1f3f5", linecolor="#dee2e6", linewidth=1, tickfont_size=11)
    return fig


# Sidebar
with st.sidebar:
    st.markdown("**Retrieval Benchmark**")
    st.markdown(
        "<span style='font-size:0.82rem;color:#6c757d;'>"
        "Chunking strategies, embedding models, and vector databases "
        "evaluated for retrieval quality."
        "</span>",
        unsafe_allow_html=True,
    )
    st.markdown("---")
    section = st.radio(
        "Navigate",
        ["Overview", "Chunking Strategy", "Embedding Models", "Vector Databases", "Raw Data"],
        label_visibility="collapsed",
    )
    st.markdown("---")
    if st.button("Reload data"):
        load_all_metrics.clear()
        st.rerun()
    st.markdown("---")
    st.markdown(
        "<span style='font-size:0.78rem;color:#868e96;'>"
        "63 docs · 45 queries<br>English / Korean / Urdu<br>"
        "5 models · 3 databases · 3 chunking strategies"
        "</span>",
        unsafe_allow_html=True,
    )


# ============================================================================
# OVERVIEW
# ============================================================================
if section == "Overview":
    st.title("Retrieval Benchmark")
    st.markdown(
        "A comparison of five embedding models and three vector databases across three chunking strategies, "
        "evaluated for semantic search quality over a multilingual corpus of 63 documents in English, Korean, and Urdu."
    )

    # Best recall across all strategies
    best = {"recall": 0, "model": "", "strategy": "", "mrr": 0}
    for strat, d in all_data.items():
        for model, mdata in d["model_axis"].items():
            r1 = mdata["overall"]["recall@1"]
            if r1 > best["recall"]:
                best = {"recall": r1, "model": model, "strategy": strat, "mrr": mdata["overall"]["mrr"]}

    ref = all_data.get("semantic", list(all_data.values())[0])
    db_data = ref["db_axis"]
    fastest_db = min(db_data, key=lambda x: db_data[x]["p50_ms"])

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(f"""<div class="metric-card-green">
            <div class="label">Best Recall@1</div>
            <div class="value">{best['recall']:.1%}</div>
            <div class="sub">{best['model']}<br>{STRATEGY_LABELS[best['strategy']]} chunking</div>
        </div>""", unsafe_allow_html=True)
    with col2:
        st.markdown(f"""<div class="metric-card-green">
            <div class="label">Best MRR</div>
            <div class="value">{best['mrr']:.3f}</div>
            <div class="sub">{best['model']}</div>
        </div>""", unsafe_allow_html=True)
    with col3:
        st.markdown(f"""<div class="metric-card-blue">
            <div class="label">Fastest DB (p50)</div>
            <div class="value">{db_data[fastest_db]['p50_ms']:.1f} ms</div>
            <div class="sub">{fastest_db}</div>
        </div>""", unsafe_allow_html=True)
    with col4:
        st.markdown(f"""<div class="metric-card-blue">
            <div class="label">Strategies Compared</div>
            <div class="value">{len(available)}</div>
            <div class="sub">{' · '.join(STRATEGY_LABELS[s] for s in available)}</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("---")

    st.subheader("Configuration variables, ranked by impact")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("**1. Chunking strategy**")
        st.markdown(
            "Sets the ceiling on recall. If a boundary splits the answer text across two chunks, "
            "no embedding model can retrieve it as a single result. "
            "Strategy choice is evaluated first because it constrains everything downstream."
        )
    with col2:
        st.markdown("**2. Embedding model**")
        st.markdown(
            "The primary driver of retrieval quality within a fixed chunking strategy. "
            "Multilingual models maintain recall across all three languages; "
            "English-only models collapse to under 10% recall on Korean and Urdu."
        )
    with col3:
        st.markdown("**3. Vector database**")
        st.markdown(
            "With exact FLAT indexes at this corpus size, all three databases return the same results. "
            "Database choice is an operational decision — latency, persistence, and infrastructure requirements "
            "rather than retrieval quality."
        )

    st.markdown("---")
    st.subheader("Corpus")
    col1, col2, col3 = st.columns(3)
    col1.metric("Documents", "63")
    col2.metric("Languages", "3")
    col3.metric("Eval Queries", "45")

    lang_df = pd.DataFrame([
        {"Language": "English", "Documents": 43, "Chunks (semantic)": 8474, "Queries": 22, "Notes": ""},
        {"Language": "Korean",  "Documents": 11, "Chunks (semantic)": 1028, "Queries": 12, "Notes": "Near-exact substring queries"},
        {"Language": "Urdu",    "Documents": 9,  "Chunks (semantic)": 954,  "Queries": 11, "Notes": "Queries draw from 3 source docs"},
    ])
    st.dataframe(lang_df, hide_index=True, use_container_width=True)
    note(
        "Korean queries are held-out sentences from the corpus, making retrieval comparatively easier for that language. "
        "Compare models within Korean rather than against English absolute scores. "
        "Urdu recall reflects concentration across 3 source documents."
    )

    st.subheader("Chunk counts by strategy")
    chunk_df = pd.DataFrame([
        {"Strategy": STRATEGY_LABELS[s], "Total Chunks": STRATEGY_CHUNKS[s]}
        for s in available
    ])
    fig = px.bar(
        chunk_df, x="Strategy", y="Total Chunks",
        text_auto=True,
        title="Total Chunks Produced per Strategy",
        color="Strategy",
        color_discrete_map={STRATEGY_LABELS[s]: STRATEGY_COLORS[s] for s in available},
    )
    clean_fig(fig)
    fig.update_layout(showlegend=False)
    st.plotly_chart(fig, use_container_width=True)
    note(
        "Fixed-size chunking produces the fewest chunks (6,091) because it packs tokens to the limit without splitting on boundaries. "
        "Semantic chunking produces the most (10,456) because natural boundaries often fall short of the token ceiling. "
        "Chunk count affects both embedding cost and the granularity of retrieval."
    )


# ============================================================================
# CHUNKING STRATEGY
# ============================================================================
elif section == "Chunking Strategy":
    st.title("Chunking Strategy Comparison")
    st.markdown(
        "Each splitting strategy produces different chunk boundaries. "
        "This section measures the effect of those boundaries on retrieval recall — "
        "the fraction of queries for which the correct passage is returned."
    )

    if len(available) < 2:
        note("Only one strategy benchmarked so far. Run report_metrics.py for the remaining strategies.", warn=True)
        st.stop()

    rows = []
    for strat in available:
        d = all_data[strat]
        for model, mdata in d["model_axis"].items():
            o = mdata["overall"]
            rows.append({
                "Strategy": STRATEGY_LABELS[strat],
                "strategy_key": strat,
                "Model": model,
                "Recall@1":  o["recall@1"],
                "Recall@5":  o["recall@5"],
                "Recall@10": o["recall@10"],
                "MRR":       o["mrr"],
            })
    df_chunk = pd.DataFrame(rows)

    st.subheader("Recall@1 by Strategy and Model")
    fig = px.bar(
        df_chunk, x="Model", y="Recall@1", color="Strategy",
        barmode="group",
        title="Recall@1 — All Models, All Chunking Strategies",
        text_auto=".2f",
        color_discrete_map={STRATEGY_LABELS[s]: STRATEGY_COLORS[s] for s in available},
    )
    clean_fig(fig)
    st.plotly_chart(fig, use_container_width=True)
    note(
        "Each group shows the same model under three different chunking strategies. "
        "A drop in recall between strategies indicates that strategy is splitting answer text "
        "across chunk boundaries more frequently for that model."
    )

    st.subheader("Recall@10 — The Retrieval Ceiling")
    st.markdown(
        "Recall@10 is the probability that the correct passage appears anywhere in the top-10 results. "
        "It is the maximum quality achievable by any downstream use of the retrieved context."
    )
    fig = px.bar(
        df_chunk, x="Model", y="Recall@10", color="Strategy",
        barmode="group",
        title="Recall@10 — All Models, All Chunking Strategies",
        text_auto=".2f",
        color_discrete_map={STRATEGY_LABELS[s]: STRATEGY_COLORS[s] for s in available},
    )
    clean_fig(fig)
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("MRR — Ranking Quality")
    st.markdown(
        "MRR (Mean Reciprocal Rank) measures where in the ranked list the correct answer appears. "
        "A result at rank 1 contributes 1.0; at rank 5 it contributes 0.2. "
        "Higher MRR means correct results surface earlier in the list."
    )
    fig = px.bar(
        df_chunk, x="Model", y="MRR", color="Strategy",
        barmode="group",
        title="MRR — All Models, All Chunking Strategies",
        text_auto=".3f",
        color_discrete_map={STRATEGY_LABELS[s]: STRATEGY_COLORS[s] for s in available},
    )
    clean_fig(fig)
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Best Model Per Strategy")
    summary_rows = []
    for strat in available:
        d = all_data[strat]
        best_model = max(d["model_axis"], key=lambda m: d["model_axis"][m]["overall"]["recall@1"])
        o = d["model_axis"][best_model]["overall"]
        summary_rows.append({
            "Strategy":  STRATEGY_LABELS[strat],
            "Chunks":    STRATEGY_CHUNKS.get(strat, ""),
            "Best Model": best_model,
            "Recall@1":  o["recall@1"],
            "Recall@5":  o["recall@5"],
            "Recall@10": o["recall@10"],
            "MRR":       o["mrr"],
        })
    st.dataframe(
        pd.DataFrame(summary_rows),
        hide_index=True,
        use_container_width=True,
        column_config={k: st.column_config.NumberColumn(format="%.3f") for k in ["Recall@1", "Recall@5", "Recall@10", "MRR"]},
    )

    st.subheader("Language Breakdown by Strategy")
    note(
        "Fixed-size chunking is language-blind — it cuts at token count regardless of sentence boundaries, "
        "which affects languages differently depending on average sentence length relative to the chunk size. "
        "Semantic chunking uses embedding similarity to find natural breaks, which tends to preserve "
        "full sentences in all three languages."
    )

    top_models = [m for m in ["bge-m3", "multilingual-e5-large"]
                  if any(m in all_data[s]["model_axis"] for s in available)]
    lang_rows = []
    for strat in available:
        d = all_data[strat]
        for model in top_models:
            if model not in d["model_axis"]:
                continue
            for lang in ["en", "ko", "ur"]:
                if lang in d["model_axis"][model]:
                    lang_rows.append({
                        "Strategy": STRATEGY_LABELS[strat],
                        "Model": model,
                        "Language": lang.upper(),
                        "Recall@1": d["model_axis"][model][lang]["recall@1"],
                    })

    if lang_rows:
        df_lang_strat = pd.DataFrame(lang_rows)
        cols = st.columns(len(top_models))
        for i, model in enumerate(top_models):
            subset = df_lang_strat[df_lang_strat["Model"] == model]
            if subset.empty:
                continue
            fig = px.bar(
                subset, x="Language", y="Recall@1", color="Strategy",
                barmode="group",
                title=model,
                text_auto=".2f",
                color_discrete_map={STRATEGY_LABELS[s]: STRATEGY_COLORS[s] for s in available},
            )
            clean_fig(fig)
            cols[i].plotly_chart(fig, use_container_width=True)

    st.subheader("Embedding Cost by Strategy")
    speed_rows = []
    for strat in available:
        for model in ["bge-large", "bge-m3", "e5-large", "multilingual-e5-large", "instructor-xl"]:
            key = (strat, model)
            if key in SPEED_DATA:
                speed_rows.append({
                    "Strategy":    STRATEGY_LABELS[strat],
                    "Model":       model,
                    "ms/chunk":    SPEED_DATA[key]["ms_per_chunk"],
                    "Total (s)":   SPEED_DATA[key]["total_s"],
                })
    df_speed = pd.DataFrame(speed_rows)

    fig = px.bar(
        df_speed, x="Model", y="ms/chunk", color="Strategy",
        barmode="group",
        title="Embedding Speed (ms/chunk) by Strategy",
        text_auto=True,
        color_discrete_map={STRATEGY_LABELS[s]: STRATEGY_COLORS[s] for s in available},
    )
    clean_fig(fig)
    st.plotly_chart(fig, use_container_width=True)
    note(
        "Fixed-size and recursive chunking produce fewer chunks than semantic, so total embedding time is lower "
        "even when ms/chunk is similar. Instructor-XL is substantially slower under all strategies "
        "(502–536 ms/chunk vs under 130 ms/chunk for all other models)."
    )


# ============================================================================
# EMBEDDING MODELS
# ============================================================================
elif section == "Embedding Models":
    st.title("Embedding Model Comparison")
    st.markdown("Retrieval quality per model, shown for a selected chunking strategy.")

    strategy_choice = st.selectbox(
        "Chunking strategy:",
        options=available,
        format_func=lambda x: STRATEGY_LABELS[x],
    )
    data = all_data[strategy_choice]

    with st.expander("Model reference"):
        speed_ref = []
        for model in ["bge-large", "bge-m3", "e5-large", "multilingual-e5-large", "instructor-xl"]:
            key = (strategy_choice, model)
            if key in SPEED_DATA:
                speed_ref.append({
                    "Model":       model,
                    "Dims":        1024 if model != "instructor-xl" else 768,
                    "Context":     8192 if model == "bge-m3" else 512,
                    "Type":        SPEED_DATA[key]["type"],
                    "ms/chunk":    SPEED_DATA[key]["ms_per_chunk"],
                    "Total time (s)": SPEED_DATA[key]["total_s"],
                    "Chunks":      STRATEGY_CHUNKS.get(strategy_choice, ""),
                })
        if speed_ref:
            st.dataframe(pd.DataFrame(speed_ref), hide_index=True, use_container_width=True)
        st.caption("Timings from GPU-backed environment (Google Colab). CPU throughput is roughly 22× slower.")

    rows = []
    for model, mdata in data["model_axis"].items():
        o = mdata["overall"]
        key = (strategy_choice, model)
        row = {
            "Model": model,
            **o,
            "type": SPEED_DATA.get(key, {}).get("type", ""),
            "ms_per_chunk": SPEED_DATA.get(key, {}).get("ms_per_chunk", 0),
        }
        for lang in ["en", "ko", "ur"]:
            row[f"r1_{lang}"] = mdata[lang]["recall@1"] if lang in mdata else None
        rows.append(row)
    df_models = pd.DataFrame(rows).sort_values("recall@1", ascending=False)

    st.subheader("Overall Retrieval Quality")
    fig = px.bar(
        df_models, x="Model", y=["recall@1", "recall@5", "recall@10"],
        barmode="group",
        title="Recall@k by Model",
        labels={"value": "Recall", "variable": "Metric"},
        text_auto=".2f",
        color_discrete_sequence=[PASTEL["blue"], PASTEL["green"], PASTEL["orange"]],
    )
    clean_fig(fig)
    st.plotly_chart(fig, use_container_width=True)
    note(
        "The gap between Recall@1 and Recall@10 shows how often the model ranks the correct chunk "
        "in the top 10 but not at position 1. A wide gap suggests the correct result is found but ranked imperfectly."
    )

    col1, col2 = st.columns(2)
    with col1:
        fig = px.bar(
            df_models, x="Model", y="mrr",
            title="Mean Reciprocal Rank",
            text_auto=".3f",
            color_discrete_sequence=[PASTEL["blue"]],
        )
        clean_fig(fig)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        fig = px.scatter(
            df_models,
            x="recall@1", y="ms_per_chunk",
            color="type", text="Model",
            title="Recall@1 vs Embedding Speed",
            labels={
                "recall@1": "Recall@1 (higher = better)",
                "ms_per_chunk": "ms/chunk (lower = better)",
                "type": "",
            },
            color_discrete_map=TYPE_COLORS,
        )
        fig.update_traces(textposition="top center", marker=dict(size=12))
        clean_fig(fig)
        st.plotly_chart(fig, use_container_width=True)

    note(
        "Instructor-XL is strictly dominated: slowest to embed and worst retrieval recall under every strategy. "
        "The multilingual models (BGE-M3, multilingual-e5-large) sit in the upper-left quadrant — "
        "better recall and faster or comparable embedding speed."
    )

    st.subheader("Precision@k")
    note(
        "With exactly one relevant chunk per query, Precision@k = Recall@k / k by definition. "
        "These numbers are artificially low — in a corpus where multiple chunks can be relevant "
        "to a single query, precision would be substantially higher."
    )
    prec_cols = [c for c in ["precision@1", "precision@3", "precision@5", "precision@10"] if c in df_models.columns]
    fig = px.bar(
        df_models[["Model"] + prec_cols], x="Model", y=prec_cols,
        barmode="group",
        title="Precision@k by Model",
        labels={"value": "Precision", "variable": "k"},
        text_auto=".2f",
        color_discrete_sequence=[PASTEL["blue"], PASTEL["green"], PASTEL["orange"], PASTEL["purple"]],
    )
    clean_fig(fig)
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Performance by Language")
    lang_rows = []
    for _, row in df_models.iterrows():
        for col_key, label in [("r1_en", "EN"), ("r1_ko", "KO"), ("r1_ur", "UR")]:
            if row.get(col_key) is not None:
                lang_rows.append({"Model": row["Model"], "Language": label, "Recall@1": row[col_key]})
    df_lang = pd.DataFrame(lang_rows)

    fig = px.bar(
        df_lang, x="Model", y="Recall@1", color="Language",
        barmode="group",
        title="Recall@1 by Model and Language",
        text_auto=".2f",
        color_discrete_map=LANG_COLORS,
    )
    clean_fig(fig)
    st.plotly_chart(fig, use_container_width=True)

    pivot = df_lang.pivot(index="Model", columns="Language", values="Recall@1")
    fig = px.imshow(
        pivot, text_auto=".2f",
        color_continuous_scale=[[0, "#f8f9fa"], [0.5, "#7BBE8A"], [1, "#2d6a4f"]],
        title="Recall@1 Heatmap — Model vs Language",
        aspect="auto",
    )
    fig.update_layout(plot_bgcolor="white", paper_bgcolor="white", margin=dict(t=45, b=10, l=10, r=10))
    st.plotly_chart(fig, use_container_width=True)

    note(
        "English-only models (bge-large, e5-large) fall to 8–9% recall on Korean and Urdu. "
        "Instructor-XL reaches 0% on both. For a multilingual corpus, only BGE-M3 and "
        "multilingual-e5-large maintain viable recall across all three languages."
    )


# ============================================================================
# VECTOR DATABASES
# ============================================================================
elif section == "Vector Databases":
    st.title("Vector Database Comparison")
    st.markdown(
        "With exact FLAT indexes, all three databases return the same top-k results. "
        "The comparison is therefore about operational characteristics: latency, build time, and memory footprint."
    )

    strategy_choice = st.selectbox(
        "Chunking strategy:",
        options=available,
        format_func=lambda x: STRATEGY_LABELS[x],
    )
    data = all_data[strategy_choice]

    with st.expander("Database reference"):
        st.markdown("""
| Database | Type | Index | Notes |
|----------|------|-------|-------|
| FAISS | In-process library | FLAT (exact) | No server, no persistence. Zero-dependency. |
| Chroma | Embedded DB | FLAT (exact) | Lightweight, persistent, simple deployment. |
| Milvus | Server (Docker) | FLAT (exact) | Distributed. Latency includes TCP round-trip. |

All three ran with exact FLAT indexes — no approximation. At this corpus size, approximate indexes are unnecessary.
        """)

    db_rows = []
    for db, ddata in data["db_axis"].items():
        db_rows.append({
            "Database":       db,
            "p50_ms":         ddata["p50_ms"],
            "p95_ms":         ddata["p95_ms"],
            "p99_ms":         ddata["p99_ms"],
            "build_seconds":  ddata["build_seconds"],
            "recall_vs_faiss": ddata["recall_vs_faiss"],
            "memory_mb":      ddata.get("index_memory_mb"),
        })
    df_db = pd.DataFrame(db_rows)

    st.subheader("Search Latency")
    fig = px.bar(
        df_db, x="Database", y=["p50_ms", "p95_ms", "p99_ms"],
        barmode="group",
        title="Search Latency Percentiles (ms)",
        labels={"value": "Latency (ms)", "variable": ""},
        text_auto=".2f",
        color_discrete_sequence=[PASTEL["blue"], PASTEL["green"], PASTEL["orange"]],
    )
    clean_fig(fig)
    st.plotly_chart(fig, use_container_width=True)
    note(
        "Milvus latency includes a TCP round-trip to Docker containers. "
        "FAISS and Chroma are in-process and do not incur this cost. "
        "Milvus tail latency (p99) is also more variable, which matters under consistent load."
    )

    col1, col2 = st.columns(2)
    with col1:
        lat_rows = []
        for db, ddata in data["db_axis"].items():
            for lat in ddata.get("search_latencies_ms", []):
                lat_rows.append({"Database": db, "Latency (ms)": lat})
        if lat_rows:
            df_lat = pd.DataFrame(lat_rows)
            fig = px.box(
                df_lat, x="Database", y="Latency (ms)",
                color="Database",
                title="Latency Distribution",
                points="outliers",
                color_discrete_map=DB_COLORS,
            )
            fig.update_layout(
                showlegend=False,
                plot_bgcolor="white", paper_bgcolor="white",
                margin=dict(t=45, b=10, l=10, r=10),
            )
            fig.update_xaxes(showgrid=False, linecolor="#dee2e6")
            fig.update_yaxes(gridcolor="#f1f3f5", linecolor="#dee2e6")
            st.plotly_chart(fig, use_container_width=True)

    with col2:
        fig = px.bar(
            df_db, x="Database", y="recall_vs_faiss",
            title="Recall vs FAISS Ground Truth",
            text_auto=".3f",
            color="Database",
            color_discrete_map=DB_COLORS,
            range_y=[0.85, 1.0],
        )
        fig.update_layout(showlegend=False)
        clean_fig(fig)
        st.plotly_chart(fig, use_container_width=True)

    note(
        "Milvus recall vs FAISS is below 1.0 even with an exact index. "
        "Disagreement only surfaces when the 10th- and 11th-best neighbors differ by less than ~2×10⁻⁵ in cosine similarity — "
        "Milvus's segment-based architecture resolves these near-ties differently than FAISS's single-pass scan. "
        "This is a tie-breaking boundary artifact, not a data integrity issue."
    )

    st.subheader("Build Time and Memory")
    col1, col2 = st.columns(2)
    with col1:
        fig = px.bar(
            df_db, x="Database", y="build_seconds",
            title="Index Build Time (s)",
            text_auto=".2f",
            color="Database",
            color_discrete_map=DB_COLORS,
        )
        fig.update_layout(showlegend=False)
        clean_fig(fig)
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        df_mem = df_db[df_db["memory_mb"].notna()].copy()
        if not df_mem.empty:
            fig = px.bar(
                df_mem, x="Database", y="memory_mb",
                title="Index Memory (MB)",
                text_auto=".1f",
                color="Database",
                color_discrete_map=DB_COLORS,
            )
            fig.update_layout(showlegend=False)
            clean_fig(fig)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Memory not available for Milvus (out-of-process).")

    st.subheader("Cross-Cut: Recall@1 by Model and Database")
    note(
        "With exact indexes, retrieval quality is determined by the embedding model, not the database. "
        "The heatmap confirms that variation across databases is negligible for the same model."
    )
    cross_rows = []
    for model in data["model_axis"]:
        for db in data["db_axis"]:
            cross_rows.append({
                "Model": model,
                "Database": db,
                "recall@1": data["crosscut"][model][db]["recall@1"],
            })
    pivot = pd.DataFrame(cross_rows).pivot(index="Model", columns="Database", values="recall@1")
    fig = px.imshow(
        pivot, text_auto=".2f",
        color_continuous_scale=[[0, "#f8f9fa"], [0.5, "#6B9DC4"], [1, "#2c5282"]],
        title="Recall@1 — Model vs Database",
        aspect="auto",
    )
    fig.update_layout(plot_bgcolor="white", paper_bgcolor="white", margin=dict(t=45, b=10, l=10, r=10))
    st.plotly_chart(fig, use_container_width=True)


# ============================================================================
# RAW DATA
# ============================================================================
elif section == "Raw Data":
    st.title("Raw Data")

    strategy_choice = st.selectbox(
        "Strategy:",
        options=available,
        format_func=lambda x: STRATEGY_LABELS[x],
    )
    data = all_data[strategy_choice]

    rows = []
    for model in data["model_axis"]:
        for db, ddata in data["db_axis"].items():
            cc = data["crosscut"][model][db]
            rows.append({
                "Model":      model,
                "DB":         db,
                "Recall@1":   cc["recall@1"],
                "Recall@5":   cc["recall@5"],
                "Recall@10":  cc["recall@10"],
                "MRR":        cc["mrr"],
                "p50_ms":     ddata["p50_ms"],
                "p95_ms":     ddata["p95_ms"],
                "p99_ms":     ddata["p99_ms"],
                "Build_s":    ddata["build_seconds"],
                "Memory_MB":  ddata.get("index_memory_mb"),
            })
    df_flat = pd.DataFrame(rows)

    col1, col2 = st.columns(2)
    with col1:
        model_filter = st.multiselect("Models", sorted(df_flat["Model"].unique()), default=sorted(df_flat["Model"].unique()))
    with col2:
        db_filter = st.multiselect("Databases", sorted(df_flat["DB"].unique()), default=sorted(df_flat["DB"].unique()))

    df_filtered = df_flat[df_flat["Model"].isin(model_filter) & df_flat["DB"].isin(db_filter)]
    st.dataframe(
        df_filtered, hide_index=True, use_container_width=True,
        column_config={
            "Recall@1":  st.column_config.NumberColumn(format="%.3f"),
            "Recall@5":  st.column_config.NumberColumn(format="%.3f"),
            "Recall@10": st.column_config.NumberColumn(format="%.3f"),
            "MRR":       st.column_config.NumberColumn(format="%.3f"),
            "p50_ms":    st.column_config.NumberColumn(format="%.2f"),
            "p95_ms":    st.column_config.NumberColumn(format="%.2f"),
            "p99_ms":    st.column_config.NumberColumn(format="%.2f"),
            "Build_s":   st.column_config.NumberColumn(format="%.2f"),
            "Memory_MB": st.column_config.NumberColumn(format="%.1f"),
        }
    )
    st.caption(f"{len(df_filtered)} rows · {STRATEGY_LABELS[strategy_choice]} chunking")

    with st.expander("Download"):
        st.download_button(
            "Download metrics.json",
            data=json.dumps(data, indent=2),
            file_name=f"metrics_{strategy_choice}.json",
            mime="application/json",
        )


# Footer
st.markdown("---")
st.markdown(
    "<span style='font-size:0.78rem;color:#868e96;'>"
    "63 documents · 5 embedding models · 3 vector databases · 3 chunking strategies · "
    "45 queries · English / Korean / Urdu"
    "</span>",
    unsafe_allow_html=True,
)