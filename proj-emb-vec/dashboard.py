# dashboard.py
# Streamlit dashboard for Embedding & Vector DB Benchmark
# Run: streamlit run dashboard.py

import json
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# --- Page config ---
st.set_page_config(
    page_title="Embedding & Vector DB Benchmark",
    page_icon="📊",
    layout="wide"
)

# --- Load data with refresh ---
@st.cache_data
def load_metrics():
    with open("results/metrics.json", "r") as f:
        return json.load(f)

# Refresh button in sidebar
with st.sidebar:
    if st.button("Refresh Data"):
        load_metrics.clear()
        st.rerun()

data = load_metrics()
meta = data["_meta"]

# --- Helper functions ---
def format_pct(x):
    return f"{x:.1%}"

def format_ms(x):
    return f"{x:.2f} ms"

# --- Sidebar ---
st.sidebar.title("Navigation")
section = st.sidebar.radio(
    "Jump to:",
    ["Overview", "Embedding Models", "Vector Databases", "Cross-Cut", "Raw Data"]
)

st.sidebar.markdown("---")
st.sidebar.caption(f"Benchmark: k={meta['k']}")
st.sidebar.caption(f"Languages: {', '.join(meta['languages'])}")
first_model = next(iter(data["model_axis"].keys()))
st.sidebar.caption(f"Queries: {data['model_axis'][first_model]['overall']['n']}")
st.sidebar.markdown("---")
st.sidebar.caption("Data from results/metrics.json")

# --- MAIN CONTENT ---

# ============================================================================
# OVERVIEW
# ============================================================================
if section == "Overview":
    st.title("Embedding & Vector Database Benchmark")
    st.markdown("A comprehensive comparison of 5 embedding models and 3 vector databases for semantic search.")

    # --- Summary cards ---
    col1, col2, col3, col4 = st.columns(4)

    # Best model
    model_data = data["model_axis"]
    best_model = max(model_data.keys(), key=lambda x: model_data[x]["overall"]["recall@1"])
    best_recall = model_data[best_model]["overall"]["recall@1"]
    col1.metric(
        "Best Embedding Model",
        best_model,
        f"Recall@1: {format_pct(best_recall)}"
    )

    # Fastest DB
    db_data = data["db_axis"]
    fastest_db = min(db_data.keys(), key=lambda x: db_data[x]["p50_ms"])
    fastest_latency = db_data[fastest_db]["p50_ms"]
    col2.metric(
        "Fastest Vector DB",
        fastest_db,
        f"p50: {format_ms(fastest_latency)}"
    )

    # Highest recall DB
    best_db = max(db_data.keys(), key=lambda x: db_data[x]["recall_vs_faiss"])
    best_db_recall = db_data[best_db]["recall_vs_faiss"]
    col3.metric(
        "Highest Recall DB",
        best_db,
        f"vs FAISS: {format_pct(best_db_recall)}"
    )

    # Best non-English
    non_en_scores = {}
    for model, mdata in model_data.items():
        scores = []
        for lang in ["ko", "ur"]:
            if lang in mdata:
                scores.append(mdata[lang]["recall@1"])
        non_en_scores[model] = sum(scores) / len(scores) if scores else 0
    best_non_en = max(non_en_scores, key=non_en_scores.get)
    col4.metric(
        "Best Non-English Model",
        best_non_en,
        f"Avg Recall@1: {format_pct(non_en_scores[best_non_en])}"
    )

    st.markdown("---")

    # --- What is this? ---
    with st.expander("What am I looking at?", expanded=True):
        st.markdown("""
        **Embedding models** convert text into vectors (lists of numbers). Similar texts have similar vectors.
        The closer two vectors are (measured by cosine similarity), the more related the texts.

        **Vector databases** store these vectors and find the nearest neighbors to a query vector quickly.
        They trade a tiny bit of accuracy for massive speed gains using approximate indexes.

        **The metrics:**
        - **Recall@k**: Of the true top-k nearest neighbors (found by exact search), what fraction did the system return?
        - **Precision@k**: Of the k results returned, how many are actually relevant?
        - **MRR** (Mean Reciprocal Rank): The average of 1/rank for the first correct result. Higher is better.
        - **p50/p95/p99**: Latency percentiles. p50 is the median; p99 is the worst 1% of requests.
        """)

    # --- Corpus stats ---
    st.subheader("Test Corpus")
    
    # Pull from data if available, otherwise use defaults (which are also fixed based on dataset)
    total_docs = data.get("_meta", {}).get("total_documents", 63)
    total_chunks = data.get("_meta", {}).get("total_chunks", 10456)
    total_queries = data["model_axis"][first_model]["overall"]["n"]
    
    col1, col2, col3 = st.columns(3)
    col1.metric("Documents", f"{total_docs:,}")
    col2.metric("Chunks", f"{total_chunks:,}")
    col3.metric("Queries", total_queries)
    
    # Language stats - dynamic if available
    lang_stats = data.get("_meta", {}).get("language_stats", {
        "en": {"docs": 43, "chunks": 8474},
        "ko": {"docs": 11, "chunks": 1028},
        "ur": {"docs": 9, "chunks": 954}
    })
    lang_df = pd.DataFrame({
        "Language": ["English", "Korean", "Urdu"],
        "Documents": [lang_stats["en"]["docs"], lang_stats["ko"]["docs"], lang_stats["ur"]["docs"]],
        "Chunks": [lang_stats["en"]["chunks"], lang_stats["ko"]["chunks"], lang_stats["ur"]["chunks"]]
    })
    st.dataframe(lang_df, hide_index=True, use_container_width=True)

    st.caption("Korean queries are held-out sentences (near-exact substrings). Compare models within Korean, not Korean vs English absolute scores.")

# ============================================================================
# EMBEDDING MODELS
# ============================================================================
elif section == "Embedding Models":
    st.title("Embedding Model Comparison")
    st.markdown("Which model retrieves the most relevant chunks?")

    # --- Model info ---
    with st.expander("About the models"):
        st.markdown("""
        | Model | Dims | Tokens | Description |
        |-------|------|--------|-------------|
        | BGE-large-en | 1024 | 512 | Strong English baseline |
        | BGE-M3 | 1024 | 8192 | Multilingual, long context |
        | E5-large-v2 | 1024 | 512 | English baseline |
        | multilingual-e5-large | 1024 | 512 | Multilingual, same family as E5 |
        | Instructor-XL | 768 | 512 | Instruction-based, large model |

        **Embedding speed** (10456 chunks):
        | Model | Time | ms/chunk |
        |-------|------|----------|
        | BGE-large-en | 1055s | 101 |
        | BGE-M3 | 975s | 93 |
        | E5-large-v2 | 1075s | 103 |
        | multilingual-e5-large | 880s | 84 |
        | Instructor-XL | 3807s | 364 |
        """)

    # --- Prepare data ---
    model_rows = []
    for model_name, mdata in data["model_axis"].items():
        row = {"Model": model_name}
        row.update(mdata["overall"])
        # Language breakdown
        for lang in ["en", "ko", "ur"]:
            if lang in mdata:
                row[f"recall@1_{lang}"] = mdata[lang]["recall@1"]
                row[f"mrr_{lang}"] = mdata[lang]["mrr"]
        model_rows.append(row)
    df_models = pd.DataFrame(model_rows)

    # Sort by recall@1
    df_models = df_models.sort_values("recall@1", ascending=False)

    # --- Overall performance ---
    st.subheader("Overall Retrieval Quality")

    fig = px.bar(
        df_models,
        x="Model",
        y=["recall@1", "recall@5", "recall@10"],
        barmode="group",
        title="Recall@k by Model",
        labels={"value": "Recall", "variable": "k"},
        text_auto=".2f"
    )
    fig.update_layout(legend_title_text="")
    st.plotly_chart(fig, use_container_width=True)

    col1, col2 = st.columns(2)

    with col1:
        fig = px.bar(
            df_models,
            x="Model",
            y="mrr",
            title="Mean Reciprocal Rank (MRR)",
            text_auto=".2f",
            color="mrr",
            color_continuous_scale="Blues"
        )
        fig.update_layout(coloraxis_showscale=False)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        fig = px.bar(
            df_models,
            x="Model",
            y="recall@1",
            title="Recall@1 (First result is correct)",
            text_auto=".2f",
            color="recall@1",
            color_continuous_scale="Blues"
        )
        fig.update_layout(coloraxis_showscale=False)
        st.plotly_chart(fig, use_container_width=True)
        
    # --- Model tradeoff: Recall vs Speed ---
    st.subheader("The Tradeoff: Recall vs Embedding Speed")

    # Embedding speed data
    speed_data = {
        "bge-large": {"ms_per_chunk": 101, "notes": "English baseline"},
        "bge-m3": {"ms_per_chunk": 93, "notes": "Multilingual, long context"},
        "e5-large": {"ms_per_chunk": 103, "notes": "English baseline"},
        "multilingual-e5-large": {"ms_per_chunk": 84, "notes": "Multilingual"},
        "instructor-xl": {"ms_per_chunk": 364, "notes": "Instruction-based"}
    }

    df_speed = df_models.copy()
    df_speed["ms_per_chunk"] = df_speed["Model"].map(lambda x: speed_data.get(x, {}).get("ms_per_chunk", 0))
    df_speed["speed_notes"] = df_speed["Model"].map(lambda x: speed_data.get(x, {}).get("notes", ""))

    fig = px.scatter(
        df_speed,
        x="recall@1",
        y="ms_per_chunk",
        color="Model",
        text="Model",
        size=[20] * len(df_speed),  # Fixed size so all dots are visible
        title="Recall@1 vs Embedding Speed (ms per chunk)",
        labels={
            "recall@1": "Recall@1 (higher = better)",
            "ms_per_chunk": "Embedding Speed (ms/chunk, lower = better)"
        },
        hover_data={"speed_notes": True}
    )
    fig.update_traces(
        textposition="top center",
        marker=dict(sizemode="diameter", sizeref=1, sizemin=12)
    )
    st.plotly_chart(fig, use_container_width=True)

    st.caption("Instructor-XL is strictly dominated: slower AND worse recall. Multilingual models offer the best speed/quality tradeoff.")    
    # --- Precision@k ---
    st.subheader("Precision@k")
    
    st.caption("Precision@k = (number of relevant results in top-k) / k. With 1 relevant chunk per query, Precision@k = Recall@k / k.")
    
    # Build precision data
    prec_cols = ["precision@1", "precision@3", "precision@5", "precision@10"]
    df_prec = df_models[["Model"] + prec_cols].copy()
    df_prec = df_prec.sort_values("precision@1", ascending=False)
    
    fig = px.bar(
        df_prec,
        x="Model",
        y=prec_cols,
        barmode="group",
        title="Precision@k by Model",
        labels={"value": "Precision", "variable": "k"},
        text_auto=".2f"
    )
    fig.update_layout(legend_title_text="")
    st.plotly_chart(fig, use_container_width=True)

    # --- Language breakdown ---
    st.subheader("Performance by Language")

    # Melt for language chart
    lang_rows = []
    for model in df_models["Model"]:
        for lang in ["en", "ko", "ur"]:
            val = df_models[df_models["Model"] == model][f"recall@1_{lang}"].values[0]
            lang_rows.append({"Model": model, "Language": lang.upper(), "Recall@1": val})
    df_lang = pd.DataFrame(lang_rows)

    fig = px.bar(
        df_lang,
        x="Model",
        y="Recall@1",
        color="Language",
        barmode="group",
        title="Recall@1 by Model and Language",
        text_auto=".2f"
    )
    st.plotly_chart(fig, use_container_width=True)

    # --- Language heatmap ---
    pivot = df_lang.pivot(index="Model", columns="Language", values="Recall@1")
    fig = px.imshow(
        pivot,
        text_auto=".2f",
        color_continuous_scale="Blues",
        title="Recall@1 Heatmap: Model vs Language",
        aspect="auto"
    )
    st.plotly_chart(fig, use_container_width=True)

    # --- Key observations ---
    with st.expander("Key observations", expanded=False):
        st.markdown("""
        - **BGE-M3** and **multilingual-e5-large** are the top performers overall, both with recall@1 ~0.55 and MRR ~0.66.
        - **Instructor-XL** performs poorly on non-English: 0% recall@1 on Korean and Urdu.
        - **BGE-large** and **E5-large** are English-only and show the expected drop on non-English.
        - The multilingual models (BGE-M3, multilingual-e5) maintain strong performance across all three languages.
        - Instructor-XL is much slower to embed (364 ms/chunk vs ~100 ms/chunk for others).
        """)

# ============================================================================
# VECTOR DATABASES
# ============================================================================
elif section == "Vector Databases":
    st.title("Vector Database Comparison")
    st.markdown("Which database is fastest and most accurate?")

    # --- DB info ---
    with st.expander("About the databases"):
        st.markdown("""
        | Database | Type | Description |
        |----------|------|-------------|
        | FAISS | Library | Exact brute-force baseline. No server, no persistence. |
        | Chroma | Embedded | Lightweight, persistent, simple API. |
        | Milvus | Server | Distributed database server (Docker). Production-scale. |

        **Key note**: All three were run with **exact (FLAT) indexes** for fair comparison.
        At 10,456 vectors, exact search is fast enough that approximate indexes aren't necessary.

        **Milvus note**: Milvus runs in Docker containers (etcd, minio, milvus). Its latency includes TCP round-trip overhead that embedded DBs don't incur.
        """)

    # --- Prepare data ---
    db_rows = []
    for db_name, ddata in data["db_axis"].items():
        row = {
            "Database": db_name,
            "p50_ms": ddata["p50_ms"],
            "p95_ms": ddata["p95_ms"],
            "p99_ms": ddata["p99_ms"],
            "build_seconds": ddata["build_seconds"],
            "recall_vs_faiss": ddata["recall_vs_faiss"],
        }
        if ddata["index_memory_mb"] is not None:
            row["memory_mb"] = ddata["index_memory_mb"]
        db_rows.append(row)
    df_db = pd.DataFrame(db_rows)

    # --- Latency ---
    st.subheader("Search Latency")

    fig = px.bar(
        df_db,
        x="Database",
        y=["p50_ms", "p95_ms", "p99_ms"],
        barmode="group",
        title="Search Latency Percentiles (ms)",
        labels={"value": "Latency (ms)", "variable": "Percentile"},
        text_auto=".2f"
    )
    fig.update_layout(legend_title_text="")
    st.plotly_chart(fig, use_container_width=True)

    col1, col2 = st.columns(2)

    with col1:
        fig = px.bar(
            df_db,
            x="Database",
            y="recall_vs_faiss",
            title="Recall vs FAISS Ground Truth",
            text_auto=".3f",
            color="recall_vs_faiss",
            color_continuous_scale="Blues",
            range_color=[0.85, 1.0]
        )
        fig.update_layout(coloraxis_showscale=False)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        fig = px.bar(
            df_db,
            x="Database",
            y="build_seconds",
            title="Index Build Time (seconds)",
            text_auto=".2f",
            color="build_seconds",
            color_continuous_scale="Blues"
        )
        fig.update_layout(coloraxis_showscale=False)
        st.plotly_chart(fig, use_container_width=True)
        
        
    # --- Latency distribution (requires raw latency data) ---
    st.subheader("Latency Distribution")

    lat_rows = []
    for db_name, ddata in data["db_axis"].items():
        if "search_latencies_ms" in ddata:
            for lat in ddata["search_latencies_ms"]:
                lat_rows.append({
                    "Database": db_name,
                    "Latency (ms)": lat
                })

    if lat_rows:
        df_lat = pd.DataFrame(lat_rows)
        fig = px.box(
            df_lat,
            x="Database",
            y="Latency (ms)",
            color="Database",
            title="Search Latency Distribution (all queries)",
            points="outliers",
            notched=True
        )
        st.plotly_chart(fig, use_container_width=True)
        st.caption("Box shows median, quartiles, and whiskers. Points show individual query latencies.")
    else:
        st.info("Raw latency data not available in metrics.json. Add 'search_latencies_ms' to each db_axis entry.")
        
    # --- Memory ---
    df_mem = df_db[df_db["memory_mb"].notna()]
    if not df_mem.empty:
        fig = px.bar(
            df_mem,
            x="Database",
            y="memory_mb",
            title="Index Memory Usage (MB)",
            text_auto=".1f",
            color="memory_mb",
            color_continuous_scale="Blues"
        )
        fig.update_layout(coloraxis_showscale=False)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Memory usage not available for Milvus (runs out-of-process).")

    # --- Tradeoff scatter ---
    st.subheader("The Tradeoff: Speed vs Accuracy")

    fig = px.scatter(
        df_db,
        x="recall_vs_faiss",
        y="p50_ms",
        size="build_seconds",
        color="Database",
        title="Latency vs Recall (point size = build time)",
        labels={
            "recall_vs_faiss": "Recall vs FAISS (higher = better)",
            "p50_ms": "Median Latency (ms, lower = better)"
        },
        text="Database"
    )
    fig.update_traces(textposition="top center")
    st.plotly_chart(fig, use_container_width=True)

    # --- Milvus deep-dive ---
    st.subheader("On Milvus Recall (0.867 vs FAISS)")

    st.markdown("""
    Milvus with a FLAT (exact) index matched FAISS ground truth precisely on well-separated similarity scores.

    **Disagreement was isolated to cases where the true top-10 boundary involved near-identical scores (within ~2e-5).**
    In these cases, Milvus's segment-based search occasionally resolved the boundary differently than FAISS's exhaustive single-pass search.

    This caused a small number of near-tied neighbors to be dropped or substituted.
    Data integrity was verified (row counts and insert success confirmed); 
    this is a narrow ranking-boundary behavior, not a data loss or scoring bug.
    """)

# ============================================================================
# CROSS-CUT
# ============================================================================
elif section == "Cross-Cut":
    st.title("Cross-Cut Analysis")
    st.markdown("How do embedding models and vector databases interact?")

    # --- Heatmap: Models x DBs = Recall@1 ---
    st.subheader("Recall@1: Model vs Database")

    cross_rows = []
    for model_name in data["model_axis"].keys():
        for db_name in data["db_axis"].keys():
            cross_rows.append({
                "Model": model_name,
                "Database": db_name,
                "recall@1": data["crosscut"][model_name][db_name]["recall@1"],
            })
    df_cross = pd.DataFrame(cross_rows)
    
    # Pivot for heatmap
    pivot = df_cross.pivot(index="Model", columns="Database", values="recall@1")

    fig = px.imshow(
        pivot,
        text_auto=".2f",
        color_continuous_scale="Blues",
        title="Recall@1: Model vs Database (exact indexes)",
        aspect="auto"
    )
    st.plotly_chart(fig, use_container_width=True)

    st.caption("With exact (FLAT) indexes, retrieval quality is determined by the embedding model, not the database. All databases return the same top-k results (modulo tie-breaking).")

    # --- Practical recommendations ---
    st.subheader("Practical Recommendations")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**For this corpus (10,456 chunks):**")
        st.markdown("""
        - **Chroma** is the fastest (1.4ms p50) with near-perfect recall (0.989)
        - **BGE-M3** or **multilingual-e5-large** give the best retrieval quality
        - **FAISS** is the simplest exact baseline
        - **Milvus** adds operational overhead without benefit at this scale
        """)

    with col2:
        st.markdown("**At larger scale (millions of chunks):**")
        st.markdown("""
        - **Milvus** is designed for distributed scaling and would become competitive
        - **Chroma** and **FAISS** are single-machine solutions
        - Approximate indexes (HNSW, IVF) would become necessary
        - The recall/latency tradeoff would shift dramatically
        """)

# ============================================================================
# RAW DATA
# ============================================================================
elif section == "Raw Data":
    st.title("Raw Data")
    
    # --- Build flat dataframe from per-query records ---
    flat_rows = []         
    for model_name in data["model_axis"].keys():
        for db_name, ddata in data["db_axis"].items():
            cc = data["crosscut"][model_name][db_name]
            flat_rows.append({
                "Model": model_name,
                "Database": db_name,
                "Recall@1": cc["recall@1"],
                "Recall@5": cc["recall@5"],
                "Recall@10": cc["recall@10"],
                "MRR": cc["mrr"],
                "p50_ms": ddata["p50_ms"],
                "p95_ms": ddata["p95_ms"],
                "p99_ms": ddata["p99_ms"],
                "Build_s": ddata["build_seconds"],
                "Memory_MB": ddata.get("index_memory_mb", None)
            })
    df_flat = pd.DataFrame(flat_rows)
    
    # --- Filters ---
    st.subheader("Filter Results")
    col1, col2 = st.columns(2)
    with col1:
        model_filter = st.multiselect(
            "Models",
            options=sorted(df_flat["Model"].unique()),
            default=sorted(df_flat["Model"].unique())
        )
    with col2:
        db_filter = st.multiselect(
            "Databases",
            options=sorted(df_flat["Database"].unique()),
            default=sorted(df_flat["Database"].unique())
        )
    
    # Apply filters
    df_filtered = df_flat[
        df_flat["Model"].isin(model_filter) &
        df_flat["Database"].isin(db_filter)
    ]
    
    # --- Data table ---
    st.dataframe(
        df_filtered,
        hide_index=True,
        use_container_width=True,
        column_config={
            "Recall@1": st.column_config.NumberColumn(format="%.3f"),
            "Recall@5": st.column_config.NumberColumn(format="%.3f"),
            "Recall@10": st.column_config.NumberColumn(format="%.3f"),
            "MRR": st.column_config.NumberColumn(format="%.3f"),
            "p50_ms": st.column_config.NumberColumn(format="%.2f"),
            "p95_ms": st.column_config.NumberColumn(format="%.2f"),
            "p99_ms": st.column_config.NumberColumn(format="%.2f"),
            "Build_s": st.column_config.NumberColumn(format="%.2f"),
            "Memory_MB": st.column_config.NumberColumn(format="%.1f"),
        }
    )
    
    st.caption(f"{len(df_filtered)} rows shown")
    
    # --- Download ---
    with st.expander("Download as JSON"):
        st.download_button(
            label="Download metrics.json",
            data=json.dumps(data, indent=2),
            file_name="metrics.json",
            mime="application/json"
        )

# ============================================================================
# FOOTER
# ============================================================================
st.markdown("---")
st.caption(f"Benchmark: k={meta['k']} | {first_model}: {data['model_axis'][first_model]['overall']['n']} queries | {', '.join(meta['languages'])} languages")
st.caption("Data from results/metrics.json")