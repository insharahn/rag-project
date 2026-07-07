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
    .callout {
        background: #f1f3f5;
        border-left: 3px solid #868e96;
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

SPEED_DATA = {
    "bge-large":             {"ms_per_chunk": 101, "type": "English-only"},
    "bge-m3":                {"ms_per_chunk": 93,  "type": "Multilingual"},
    "e5-large":              {"ms_per_chunk": 103, "type": "English-only"},
    "multilingual-e5-large": {"ms_per_chunk": 84,  "type": "Multilingual"},
    "instructor-xl":         {"ms_per_chunk": 364, "type": "Instruction-based"},
}

COLORS = {
    "primary":   "#212529",
    "secondary": "#868e96",
    "light":     "#ced4da",
    "three":     ["#212529", "#868e96", "#ced4da"],
    "two":       ["#212529", "#adb5bd"],
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


def clean_fig(fig, show_x_grid=False):
    fig.update_layout(
        plot_bgcolor="white",
        paper_bgcolor="white",
        font_family="sans-serif",
        font_size=12,
        margin=dict(t=40, b=30, l=10, r=10),
        legend=dict(bgcolor="rgba(0,0,0,0)", borderwidth=0),
    )
    fig.update_xaxes(showgrid=show_x_grid, linecolor="#dee2e6", linewidth=1, tickfont_size=11)
    fig.update_yaxes(gridcolor="#f1f3f5", linecolor="#dee2e6", linewidth=1, tickfont_size=11)
    return fig


# Sidebar
with st.sidebar:
    st.markdown("**Retrieval Benchmark**")
    st.markdown(
        "<span style='font-size:0.82rem;color:#6c757d;'>"
        "Chunking strategies, embedding models, and vector databases "
        "evaluated for RAG retrieval quality."
        "</span>",
        unsafe_allow_html=True,
    )
    st.markdown("---")
    section = st.radio(
        "Navigate",
        ["Overview", "Chunking Strategy", "Embedding Models", "Vector Databases", "RAG Recommendations", "Raw Data"],
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
        "This benchmark identifies the best chunking strategy, embedding model, and vector database "
        "for semantic retrieval over a multilingual corpus. All results are evaluated against the "
        "same 45 queries to support direct comparison across configurations."
    )

    # Find overall best recall@1 across all strategies
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
        st.markdown(f"""<div class="metric-card">
            <div class="label">Best Recall@1</div>
            <div class="value">{best['recall']:.1%}</div>
            <div class="sub">{best['model']}<br>{STRATEGY_LABELS[best['strategy']]} chunking</div>
        </div>""", unsafe_allow_html=True)
    with col2:
        st.markdown(f"""<div class="metric-card">
            <div class="label">Best MRR</div>
            <div class="value">{best['mrr']:.3f}</div>
            <div class="sub">{best['model']}</div>
        </div>""", unsafe_allow_html=True)
    with col3:
        st.markdown(f"""<div class="metric-card">
            <div class="label">Fastest DB (p50)</div>
            <div class="value">{db_data[fastest_db]['p50_ms']:.1f} ms</div>
            <div class="sub">{fastest_db}</div>
        </div>""", unsafe_allow_html=True)
    with col4:
        st.markdown(f"""<div class="metric-card">
            <div class="label">Strategies Compared</div>
            <div class="value">{len(available)}</div>
            <div class="sub">{' · '.join(STRATEGY_LABELS[s] for s in available)}</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("---")

    st.subheader("Three Variables, One Pipeline")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("**1. Chunking strategy**")
        st.markdown(
            "The highest-leverage variable. A boundary that splits the answer sentence across two chunks "
            "makes that answer irretrievable regardless of model quality. Strategy choice sets the ceiling on recall."
        )
    with col2:
        st.markdown("**2. Embedding model**")
        st.markdown(
            "The second-highest variable. Multilingual models (BGE-M3, multilingual-e5-large) maintain strong "
            "recall across all three languages. English-only models collapse to under 10% recall on Korean and Urdu."
        )
    with col3:
        st.markdown("**3. Vector database**")
        st.markdown(
            "An operational choice, not a retrieval one. With exact FLAT indexes, all three databases return "
            "the same results. Choose based on latency, persistence, and deployment constraints."
        )

    st.markdown("---")
    st.subheader("Corpus")

    col1, col2, col3 = st.columns(3)
    col1.metric("Documents", "63")
    col2.metric("Languages", "3")
    col3.metric("Eval Queries", "45")

    lang_df = pd.DataFrame([
        {"Language": "English", "Documents": 43, "Queries": 22, "Notes": ""},
        {"Language": "Korean",  "Documents": 11, "Queries": 12, "Notes": "Near-exact substring queries"},
        {"Language": "Urdu",    "Documents": 9,  "Queries": 11, "Notes": "Queries draw from 3 source docs"},
    ])
    st.dataframe(lang_df, hide_index=True, use_container_width=True)
    note(
        "Korean queries are held-out sentences from the corpus, not natural conversational questions. "
        "Korean recall is comparatively easier — compare models within Korean, not Korean vs English absolute scores. "
        "Urdu's strong scores reflect concentration in 3 documents, not a general signal about model quality."
    )

    st.markdown("---")
    st.subheader("What the metrics mean for a chatbot")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
**Recall@k** is the primary RAG metric. If the correct chunk isn't in the top-k results passed to the LLM, no amount of prompting recovers it. Recall@10 is the hard ceiling on answer quality.

**MRR** (Mean Reciprocal Rank) reflects ranking quality — a correct answer at position 1 is more valuable than one at position 10 because earlier context receives more attention weight.
        """)
    with col2:
        st.markdown("""
**Recall@1 vs Recall@10** — the gap between them shows how often the model *knows* the answer is nearby but ranks it imperfectly. A large gap means re-ranking or larger context windows would help.

**Search latency** adds directly to chatbot response time. At scale with thousands of daily queries, the difference between 1ms and 15ms per retrieval compounds significantly.
        """)


# ============================================================================
# CHUNKING STRATEGY
# ============================================================================
elif section == "Chunking Strategy":
    st.title("Chunking Strategy Comparison")
    st.markdown(
        "Before any embedding model can retrieve a passage, that passage must exist as a single intact chunk. "
        "This section measures how each splitting strategy affects retrieval recall across all models and languages."
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
                "Recall@1": o["recall@1"],
                "Recall@5": o["recall@5"],
                "Recall@10": o["recall@10"],
                "MRR": o["mrr"],
            })
    df_chunk = pd.DataFrame(rows)

    st.subheader("Recall@1 by Strategy and Model")
    fig = px.bar(
        df_chunk, x="Model", y="Recall@1", color="Strategy",
        barmode="group",
        title="Recall@1 — All Models, All Chunking Strategies",
        text_auto=".2f",
        color_discrete_sequence=COLORS["three"],
    )
    clean_fig(fig)
    st.plotly_chart(fig, use_container_width=True)
    note(
        "Each bar group shows the same model under different chunking strategies. "
        "A drop in recall between strategies means that strategy is splitting answer text across boundaries more often."
    )

    st.subheader("Recall@10 — The RAG Ceiling")
    st.markdown(
        "Recall@10 is the probability the LLM sees the answer at all when given 10 chunks. "
        "This is the hard ceiling on RAG answer quality, independent of how good the LLM is."
    )
    fig = px.bar(
        df_chunk, x="Model", y="Recall@10", color="Strategy",
        barmode="group",
        title="Recall@10 — All Models, All Chunking Strategies",
        text_auto=".2f",
        color_discrete_sequence=COLORS["three"],
    )
    clean_fig(fig)
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("MRR — Rank Quality")
    fig = px.bar(
        df_chunk, x="Model", y="MRR", color="Strategy",
        barmode="group",
        title="Mean Reciprocal Rank — All Models, All Chunking Strategies",
        text_auto=".3f",
        color_discrete_sequence=COLORS["three"],
    )
    clean_fig(fig)
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Strategy Summary — Best Model Per Strategy")
    summary_rows = []
    for strat in available:
        d = all_data[strat]
        best_model = max(d["model_axis"], key=lambda m: d["model_axis"][m]["overall"]["recall@1"])
        o = d["model_axis"][best_model]["overall"]
        summary_rows.append({
            "Strategy": STRATEGY_LABELS[strat],
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
    st.markdown(
        "Chunking interacts with language differently. Fixed-size splitting is blind to sentence boundaries "
        "in any language. Semantic chunking uses embedding similarity to find natural breaks, which helps "
        "particularly for Urdu and Korean where sentence markers differ from English."
    )

    top_models = [m for m in ["bge-m3", "multilingual-e5-large"] if any(m in all_data[s]["model_axis"] for s in available)]
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
                title=f"{model}",
                text_auto=".2f",
                color_discrete_sequence=COLORS["three"],
            )
            clean_fig(fig)
            cols[i].plotly_chart(fig, use_container_width=True)


# ============================================================================
# EMBEDDING MODELS
# ============================================================================
elif section == "Embedding Models":
    st.title("Embedding Model Comparison")
    st.markdown("Which model produces representations that retrieve the most relevant chunks? Results shown per chunking strategy.")

    strategy_choice = st.selectbox(
        "Chunking strategy:",
        options=available,
        format_func=lambda x: STRATEGY_LABELS[x],
    )
    data = all_data[strategy_choice]

    with st.expander("Model reference"):
        st.markdown("""
| Model | Dims | Context | Type | ms/chunk (GPU) |
|-------|------|---------|------|----------------|
| bge-large | 1024 | 512 | English-only | 101 |
| bge-m3 | 1024 | 8,192 | Multilingual | 93 |
| e5-large | 1024 | 512 | English-only | 103 |
| multilingual-e5-large | 1024 | 512 | Multilingual | 84 |
| instructor-xl | 768 | 512 | Instruction-based | 364 |

CPU throughput is roughly 22× slower than the GPU figures above.
        """)

    rows = []
    for model, mdata in data["model_axis"].items():
        o = mdata["overall"]
        row = {"Model": model, **o,
               "type": SPEED_DATA.get(model, {}).get("type", ""),
               "ms_per_chunk": SPEED_DATA.get(model, {}).get("ms_per_chunk", 0)}
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
        color_discrete_sequence=COLORS["three"],
    )
    clean_fig(fig)
    st.plotly_chart(fig, use_container_width=True)

    col1, col2 = st.columns(2)
    with col1:
        fig = px.bar(
            df_models, x="Model", y="mrr",
            title="Mean Reciprocal Rank",
            text_auto=".3f",
            color_discrete_sequence=[COLORS["primary"]],
        )
        clean_fig(fig)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        color_map = {"Multilingual": COLORS["primary"], "English-only": COLORS["secondary"], "Instruction-based": COLORS["light"]}
        fig = px.scatter(
            df_models,
            x="recall@1", y="ms_per_chunk",
            color="type", text="Model",
            title="Recall@1 vs Embedding Speed",
            labels={"recall@1": "Recall@1 (higher = better)", "ms_per_chunk": "ms/chunk (lower = better)", "type": ""},
            color_discrete_map=color_map,
        )
        fig.update_traces(textposition="top center", marker=dict(size=12))
        clean_fig(fig)
        st.plotly_chart(fig, use_container_width=True)

    note(
        "Instructor-XL is strictly dominated under every chunking strategy: "
        "slowest to embed (364ms/chunk, 4× slower than others) and worst retrieval recall. "
        "There is no configuration in which it is the right choice for this corpus."
    )

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
        color_discrete_sequence=COLORS["three"],
    )
    clean_fig(fig)
    st.plotly_chart(fig, use_container_width=True)

    pivot = df_lang.pivot(index="Model", columns="Language", values="Recall@1")
    fig = px.imshow(
        pivot, text_auto=".2f",
        color_continuous_scale=[[0, "#f8f9fa"], [0.5, "#868e96"], [1, "#212529"]],
        title="Recall@1 Heatmap — Model vs Language",
        aspect="auto",
    )
    fig.update_layout(plot_bgcolor="white", paper_bgcolor="white", margin=dict(t=40, b=10, l=10, r=10))
    st.plotly_chart(fig, use_container_width=True)

    note(
        "English-only models (bge-large, e5-large) fall to 8–9% recall on Korean and Urdu. "
        "Instructor-XL reaches 0% on both. For a multilingual chatbot, only BGE-M3 and "
        "multilingual-e5-large are viable — the others will fail silently on non-English queries."
    )

    st.subheader("Precision@k")
    note(
        "With exactly one relevant chunk per query, Precision@k = Recall@k / k. "
        "Precision is artificially low here. In a real RAG corpus where multiple chunks can be relevant, "
        "these numbers would be substantially higher."
    )
    prec_cols = ["precision@1", "precision@3", "precision@5", "precision@10"]
    df_prec = df_models[["Model"] + [c for c in prec_cols if c in df_models.columns]].copy()
    fig = px.bar(
        df_prec, x="Model", y=prec_cols,
        barmode="group",
        title="Precision@k by Model",
        labels={"value": "Precision", "variable": "k"},
        text_auto=".2f",
        color_discrete_sequence=COLORS["three"] + ["#f1f3f5"],
    )
    clean_fig(fig)
    st.plotly_chart(fig, use_container_width=True)


# ============================================================================
# VECTOR DATABASES
# ============================================================================
elif section == "Vector Databases":
    st.title("Vector Database Comparison")
    st.markdown(
        "With exact FLAT indexes, the database does not affect retrieval recall — "
        "all three return the same top-k results. This section covers the operational "
        "dimensions: latency, build time, and memory."
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
| FAISS | In-process library | FLAT (exact) | No server, no persistence. Zero-dependency baseline. |
| Chroma | Embedded DB | FLAT (exact) | Lightweight, persistent. Simple deployment. |
| Milvus | Server (Docker) | FLAT (exact) | Distributed. Latency includes TCP round-trip. |

All three ran with exact FLAT indexes — no approximation. At ~10k vectors, approximate indexes (HNSW, IVF) are unnecessary.
        """)

    db_rows = []
    for db, ddata in data["db_axis"].items():
        db_rows.append({
            "Database": db,
            "p50_ms": ddata["p50_ms"],
            "p95_ms": ddata["p95_ms"],
            "p99_ms": ddata["p99_ms"],
            "build_seconds": ddata["build_seconds"],
            "recall_vs_faiss": ddata["recall_vs_faiss"],
            "memory_mb": ddata.get("index_memory_mb"),
        })
    df_db = pd.DataFrame(db_rows)

    st.subheader("Search Latency")
    fig = px.bar(
        df_db, x="Database", y=["p50_ms", "p95_ms", "p99_ms"],
        barmode="group",
        title="Search Latency Percentiles (ms)",
        labels={"value": "Latency (ms)", "variable": ""},
        text_auto=".2f",
        color_discrete_sequence=COLORS["three"],
    )
    clean_fig(fig)
    st.plotly_chart(fig, use_container_width=True)
    note(
        "Milvus latency includes a TCP round-trip to Docker. "
        "FAISS and Chroma are in-process and don't pay this cost. "
        "In a bare-metal production deployment, Milvus latency would be lower."
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
                color_discrete_sequence=COLORS["three"],
            )
            fig.update_layout(
                showlegend=False,
                plot_bgcolor="white", paper_bgcolor="white",
                margin=dict(t=40, b=10, l=10, r=10),
            )
            fig.update_xaxes(showgrid=False, linecolor="#dee2e6")
            fig.update_yaxes(gridcolor="#f1f3f5", linecolor="#dee2e6")
            st.plotly_chart(fig, use_container_width=True)

    with col2:
        fig = px.bar(
            df_db, x="Database", y="recall_vs_faiss",
            title="Recall vs FAISS Ground Truth",
            text_auto=".3f",
            color_discrete_sequence=[COLORS["primary"]],
            range_y=[0.85, 1.0],
        )
        clean_fig(fig)
        st.plotly_chart(fig, use_container_width=True)

    note(
        "Milvus recall vs FAISS is below 1.0 even with an exact index. "
        "Disagreement only surfaces when the true 10th- and 11th-best neighbors differ by less than ~2×10⁻⁵ in cosine similarity. "
        "Milvus's segment-based architecture resolves these near-ties differently than FAISS's single-pass scan. "
        "This is a tie-breaking artifact, not a data integrity issue."
    )

    st.subheader("Build Time and Memory")
    col1, col2 = st.columns(2)
    with col1:
        fig = px.bar(
            df_db, x="Database", y="build_seconds",
            title="Index Build Time (s)",
            text_auto=".2f",
            color_discrete_sequence=[COLORS["primary"]],
        )
        clean_fig(fig)
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        df_mem = df_db[df_db["memory_mb"].notna()].copy()
        if not df_mem.empty:
            fig = px.bar(
                df_mem, x="Database", y="memory_mb",
                title="Index Memory (MB)",
                text_auto=".1f",
                color_discrete_sequence=[COLORS["primary"]],
            )
            clean_fig(fig)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Memory not available for Milvus (out-of-process).")

    st.subheader("Cross-Cut: Recall@1 by Model and Database")
    note("Confirms that with exact indexes, retrieval quality is model-determined. Variation across databases is negligible.")
    cross_rows = []
    for model in data["model_axis"]:
        for db in data["db_axis"]:
            cross_rows.append({
                "Model": model,
                "Database": db,
                "recall@1": data["crosscut"][model][db]["recall@1"],
            })
    df_cross = pd.DataFrame(cross_rows)
    pivot = df_cross.pivot(index="Model", columns="Database", values="recall@1")
    fig = px.imshow(
        pivot, text_auto=".2f",
        color_continuous_scale=[[0, "#f8f9fa"], [0.5, "#868e96"], [1, "#212529"]],
        title="Recall@1 — Model vs Database",
        aspect="auto",
    )
    fig.update_layout(plot_bgcolor="white", paper_bgcolor="white", margin=dict(t=40, b=10, l=10, r=10))
    st.plotly_chart(fig, use_container_width=True)


# ============================================================================
# RAG RECOMMENDATIONS
# ============================================================================
elif section == "RAG Recommendations":
    st.title("RAG Stack Recommendations")
    st.markdown(
        "Given that this benchmark feeds directly into a chatbot's retrieval pipeline, "
        "recommendations are framed around what produces the best answers at the end of the RAG chain, "
        "not just the best recall numbers in isolation."
    )

    st.subheader("Decision Order")
    st.markdown("""
The three configuration choices, ranked by impact on retrieval quality:

**1. Chunking strategy** — sets the ceiling. A boundary that splits an answer across two chunks makes that answer irretrievable. Get this right first.

**2. Embedding model** — determines whether non-English queries work at all. English-only models are not viable for a multilingual chatbot.

**3. Vector database** — an operational choice. Does not affect recall with exact indexes at this corpus size. Choose based on latency, persistence, and infrastructure constraints.
    """)

    st.markdown("---")
    st.subheader("Configurations")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Best overall retrieval quality**")
        st.markdown("""
- Chunking: whichever strategy produced highest recall@1 in your results
- Model: BGE-M3 (best multilingual recall, 8192-token context window handles longer semantic chunks without truncation)
- DB: Chroma (lowest latency, near-perfect recall vs FAISS, no server required)
        """)
        st.markdown("**Fastest to embed, close recall**")
        st.markdown("""
- Chunking: recursive (good boundary quality, approximately 7× faster to run than semantic)
- Model: multilingual-e5-large (84ms/chunk vs 93ms/chunk for BGE-M3, within a small margin on recall)
- DB: Chroma
        """)
        st.markdown("**Zero-dependency deployment**")
        st.markdown("""
- Model: BGE-M3
- DB: FAISS (no server, no Docker, in-process, exact search)
- Note: FAISS is slower than Chroma on this corpus but removes all infrastructure requirements
        """)

    with col2:
        st.markdown("**What to avoid**")
        st.markdown("""
- **Instructor-XL**: 364ms/chunk, 0% recall on Korean and Urdu, worst overall recall. Strictly dominated — no compensating advantage exists.
- **English-only models**: will silently fail on non-English user queries. Fine if your chatbot is English-only, wrong otherwise.
- **Milvus at this scale**: adds Docker infrastructure and TCP latency with no recall benefit. Not worth it for a single-machine deployment under 100k chunks.
        """)
        st.markdown("**At larger scale**")
        st.markdown("""
- Milvus becomes the correct DB choice when horizontal scaling is required
- Approximate indexes (HNSW, IVF) become necessary — expect recall@k to drop as corpus grows
- Chunking strategy becomes more critical at scale as the number of near-identical chunks increases
- Consider re-ranking (cross-encoder) on top of retrieval to improve MRR without changing recall@10
        """)

    st.markdown("---")
    st.subheader("Full Recall Reference")
    note("Recall@10 is the ceiling — the LLM can only answer from the context it receives. If the correct chunk isn't in the top 10, no prompting recovers it.")

    ref_rows = []
    for strat in available:
        d = all_data[strat]
        for model in ["bge-m3", "multilingual-e5-large", "bge-large", "e5-large", "instructor-xl"]:
            if model not in d["model_axis"]:
                continue
            o = d["model_axis"][model]["overall"]
            ref_rows.append({
                "Strategy": STRATEGY_LABELS[strat],
                "Model": model,
                "Recall@1":  o["recall@1"],
                "Recall@5":  o["recall@5"],
                "Recall@10": o["recall@10"],
                "MRR":       o["mrr"],
            })
    st.dataframe(
        pd.DataFrame(ref_rows),
        hide_index=True,
        use_container_width=True,
        column_config={k: st.column_config.NumberColumn(format="%.3f") for k in ["Recall@1", "Recall@5", "Recall@10", "MRR"]},
    )

    st.markdown("---")
    st.subheader("Limitations")
    st.markdown("""
- **Single relevant chunk per query.** Precision@k is artificially low. In a real RAG corpus with multiple relevant passages, these numbers would be higher.
- **Korean queries are held-out sentences**, not natural questions. Korean scores are comparatively easier and should not be compared directly against English absolute numbers.
- **Urdu queries draw from 3 source documents.** Urdu recall reflects concentration in a few documents, not generalization across the language.
- **Latency measured on a single laptop with Docker running.** Production server numbers, especially for Milvus on dedicated hardware, would differ.
- **Exact indexes only.** At 1M+ chunks, approximate indexes become necessary and the recall/latency story changes substantially.
    """)


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
                "Model": model, "DB": db,
                "Recall@1": cc["recall@1"], "Recall@5": cc["recall@5"], "Recall@10": cc["recall@10"], "MRR": cc["mrr"],
                "p50_ms": ddata["p50_ms"], "p95_ms": ddata["p95_ms"], "p99_ms": ddata["p99_ms"],
                "Build_s": ddata["build_seconds"],
                "Memory_MB": ddata.get("index_memory_mb"),
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
    st.caption(f"{len(df_filtered)} rows · strategy: {STRATEGY_LABELS[strategy_choice]}")

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