#
# Author: <YOUR_FULL_NAME>  |  Index: <YOUR_INDEX_NUMBER>
# ---------------------------------------------------------------------------
# app.py — Streamlit UI for the manual RAG pipeline (CS4241 / Academic City).
#
# Run from the ``ai_1`` project folder::
#
#     streamlit run app.py
#
# This file only orchestrates your existing ``src.*`` modules; it does not
# implement retrieval or generation logic itself.
#


from __future__ import annotations

import os
import sys
import traceback
from pathlib import Path
from typing import Any
from dotenv import load_dotenv

load_dotenv()

# -----------------------------------------------------------------------------
# Path setup (so ``import src...`` works when Streamlit launches this file)
# -----------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import streamlit as st

from src.chunking.chunker import (
    chunks_to_jsonable,
    csv_rows_to_chunks,
    pdf_pages_to_chunks,
)
from src.embedding.embedder import DEFAULT_MODEL_NAME, SentenceTransformerEmbedder
from src.generation.llm_client import LLMResponse, OpenAIChatClient
from src.generation.prompt_builder import (
    ContextAssemblyConfig,
    PromptStyle,
    build_rag_prompt,
)
from src.ingest.load_csv import load_csv_as_documents
from src.ingest.load_pdf import load_pdf_document
from src.logging.logger import RagJsonLogger
from src.retrieval.faiss_index import MANIFEST_FILENAME, FaissChunkIndex
from src.retrieval.hybrid_ranker import HybridConfig, HybridRetriever
from src.retrieval.retriever import VectorRetrievalConfig, VectorRetriever
from src.utils.helpers import load_json, save_json


# --- Paths (exam layout) ------------------------------------------------------
RAW_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
FAISS_DIR = PROCESSED_DIR / "faiss_store"
CHUNKS_JSON = PROCESSED_DIR / "chunks.json"
CSV_FILENAME = "Ghana_Election_Result.csv"


def _init_session_state() -> None:
    defaults: dict[str, Any] = {
        "index_cache_bust": 0,
        "last_session_paths": None,
        "last_errors": [],
        "stage_status": {},
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


def _set_stage(name: str, state: str, detail: str = "") -> None:
    st.session_state.setdefault("stage_status", {})
    st.session_state["stage_status"][name] = {"state": state, "detail": detail}


def resolve_pdf_path(raw_dir: Path) -> Path | None:
        # Pick first usable budget-like PDF under ``data/raw``.

    candidates: list[Path] = []
    for name in ("2025_budget.pdf", "budget_2025.pdf"):
        candidates.append(raw_dir / name)
    candidates.extend(sorted(raw_dir.glob("*budget*.pdf")))
    candidates.extend(sorted(raw_dir.glob("*.pdf")))
    seen: set[str] = set()
    for p in candidates:
        key = str(p.resolve())
        if key in seen:
            continue
        seen.add(key)
        if p.is_file() and p.stat().st_size > 0:
            return p
    return None


@st.cache_resource(show_spinner="Loading sentence-transformers model…")
def cached_embedder(model_name: str) -> SentenceTransformerEmbedder:
    # Heavy model load — cached by model id string.
    return SentenceTransformerEmbedder(model_name=model_name, default_batch_size=32)


@st.cache_resource(show_spinner="Loading FAISS index from disk…")
def cached_faiss_index(faiss_dir_str: str, cache_bust: int) -> FaissChunkIndex:
    # Reload index when ``cache_bust`` changes (after rebuild).
    # Args:
    # faiss_dir_str: String path so Streamlit hashing stays predictable.
    # cache_bust: Integer bumped in session state after each rebuild.

    return FaissChunkIndex.load(Path(faiss_dir_str))


@st.cache_data(show_spinner=False)
def load_chunks_from_manifest(manifest_path_str: str, mtime: float) -> list[dict[str, Any]]:
    # Chunk list for hybrid keyword pass (mirrors FAISS manifest order).
    _ = mtime  # file content tied to mtime for cache invalidation
    data = load_json(Path(manifest_path_str))
    chunks = data.get("chunks")
    if not isinstance(chunks, list):
        return []
    return [dict(c) for c in chunks]


def load_and_prepare_documents(
    *,
    use_csv: bool,
    use_pdf: bool,
    chunk_size: int,
    overlap: int,
) -> tuple[list[dict[str, Any]], list[str]]:
    # Load raw sources and produce chunk dicts (same shape as FAISS manifest).
    # Returns:
    # ``(chunks_as_jsonable_dicts, warnings_or_errors)``

    notes: list[str] = []
    all_chunks: list[Any] = []

    if use_csv:
        csv_path = RAW_DIR / CSV_FILENAME
        if not csv_path.is_file():
            notes.append(f"CSV not found: {csv_path}")
        else:
            try:
                _, docs = load_csv_as_documents(str(csv_path))
                ch = csv_rows_to_chunks(
                    docs, chunk_size=chunk_size, overlap=overlap, apply_cleaning=True
                )
                all_chunks.extend(ch)
                notes.append(f"Loaded CSV: {len(docs)} rows → {len(ch)} chunks.")
            except Exception as exc:  # noqa: BLE001
                notes.append(f"CSV load failed: {exc}")

    if use_pdf:
        pdf_path = resolve_pdf_path(RAW_DIR)
        if pdf_path is None:
            notes.append("No non-empty PDF found under data/raw — skipped PDF.")
        else:
            try:
                pages = load_pdf_document(
                    pdf_path, apply_cleaning=True, output_json=None
                )
                if not pages:
                    notes.append("PDF produced zero pages of text — skipped.")
                else:
                    ch = pdf_pages_to_chunks(
                        pages,
                        chunk_size=chunk_size,
                        overlap=overlap,
                        apply_cleaning=False,
                    )
                    all_chunks.extend(ch)
                    notes.append(
                        f"Loaded PDF: {pdf_path.name} — {len(pages)} pages → {len(ch)} chunks."
                    )
            except Exception as exc:  # noqa: BLE001
                notes.append(f"PDF load failed: {exc}")

    if not all_chunks:
        return [], notes

    return chunks_to_jsonable(all_chunks), notes


def build_faiss_from_chunks(
    chunk_dicts: list[dict[str, Any]],
    embedder: SentenceTransformerEmbedder,
) -> FaissChunkIndex:
        # Embed all chunks and build a fresh ``IndexFlatIP`` store.

    batch = embedder.embed_chunks(chunk_dicts)
    return FaissChunkIndex.build(batch.vectors, chunk_dicts)


def build_or_load_index(
    *,
    rebuild: bool,
    use_csv: bool,
    use_pdf: bool,
    chunk_size: int,
    overlap: int,
    embedder_model: str,
) -> tuple[FaissChunkIndex | None, list[dict[str, Any]], list[str]]:
    # Either rebuild embeddings + FAISS or load existing ``faiss_store``.
    # Returns:
    # ``(index_or_none, chunk_dicts, messages)``

    messages: list[str] = []
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    manifest_path = FAISS_DIR / MANIFEST_FILENAME

    if (
        not rebuild
        and manifest_path.is_file()
        and FAISS_DIR.joinpath("faiss_index_flatip.faiss").is_file()
    ):
        try:
            mtime = manifest_path.stat().st_mtime
            idx = FaissChunkIndex.load(FAISS_DIR)
            chunks = load_chunks_from_manifest(str(manifest_path), mtime)
            messages.append(
                f"Reused existing index ({idx.ntotal} vectors) from {FAISS_DIR}."
            )
            return idx, chunks, messages
        except Exception as exc:  # noqa: BLE001
            messages.append(f"Failed to load existing index, will rebuild: {exc}")

    chunk_dicts, prep_notes = load_and_prepare_documents(
        use_csv=use_csv,
        use_pdf=use_pdf,
        chunk_size=chunk_size,
        overlap=overlap,
    )
    messages.extend(prep_notes)

    if not chunk_dicts:
        messages.append("No chunks produced — cannot build index.")
        return None, [], messages

    try:
        embedder = cached_embedder(embedder_model)
    except Exception as exc:  # noqa: BLE001
        messages.append(f"Embedder failed: {exc}\n{traceback.format_exc()}")
        return None, [], messages

    try:
        index = build_faiss_from_chunks(chunk_dicts, embedder)
        FAISS_DIR.mkdir(parents=True, exist_ok=True)
        index.save(FAISS_DIR)
        save_json(chunk_dicts, CHUNKS_JSON)
        messages.append(f"Built and saved FAISS index ({index.ntotal} vectors).")
    except Exception as exc:  # noqa: BLE001
        messages.append(f"Index build failed: {exc}\n{traceback.format_exc()}")
        return None, [], messages

    st.session_state["index_cache_bust"] = int(st.session_state.get("index_cache_bust", 0)) + 1
    return index, chunk_dicts, messages


def run_retrieval(
    *,
    mode: str,
    embedder: SentenceTransformerEmbedder,
    index: FaissChunkIndex,
    all_chunk_dicts: list[dict[str, Any]],
    query: str,
    top_k: int,
    hybrid_alpha: float,
    hybrid_beta: float,
) -> tuple[list[dict[str, Any]], str]:
        # Vector-only or hybrid retrieval; returns hits + mode label.

    pool = max(int(top_k), 10)
    vr = VectorRetriever(
        embedder,
        index,
        config=VectorRetrievalConfig(top_k=pool),
    )
    if mode == "vector":
        hits = vr.retrieve(query, top_k=top_k, include_score_unit_interval=True)
        return hits, "vector"

    hy = HybridRetriever(
        vr,
        all_chunks=all_chunk_dicts,
        config=HybridConfig(
            alpha=float(hybrid_alpha),
            beta=float(hybrid_beta),
            vector_pool_k=max(20, top_k),
            keyword_pool_k=30,
            final_top_k=int(top_k),
        ),
    )
    hits = hy.retrieve(query, top_k=top_k)
    return hits, "hybrid"


def run_generation(
    *,
    api_messages: list[dict[str, str]],
    temperature: float,
    max_tokens: int,
    model_override: str,
) -> tuple[LLMResponse | None, str | None]:
        # Call OpenAI-compatible chat API; returns (response, error_message).

    if not os.environ.get("OPENAI_API_KEY", "").strip():
        return None, "OPENAI_API_KEY is not set — retrieval still works; generation is disabled."

    try:
        client = OpenAIChatClient(
            model=model_override.strip() or None,
        )
        return client.chat(
            api_messages,
            temperature=float(temperature),
            max_tokens=int(max_tokens) if max_tokens > 0 else None,
        ), None
    except Exception as exc:  # noqa: BLE001
        return None, str(exc)


def render_hit(hit: dict[str, Any], rank: int) -> None:
        # One expandable row for retrieval transparency.

    score = hit.get("final_score", hit.get("score", hit.get("score_01", 0.0)))
    try:
        score_s = f"{float(score):.4f}"
    except (TypeError, ValueError):
        score_s = str(score)
    cid = hit.get("chunk_id", "(no id)")
    meta = hit.get("metadata") or {}
    title = f"Rank {rank} · {cid} · score={score_s}"
    with st.expander(title):
        st.markdown("**Source metadata**")
        st.json(meta, expanded=False)
        st.markdown("**Chunk text**")
        st.text_area("text", value=str(hit.get("text", "")), height=220, key=f"tx_{rank}_{cid}", disabled=True)


def _hits_for_log(hits: list[dict[str, Any]], max_text: int = 800) -> list[dict[str, Any]]:
        # Trim hit payloads for JSONL size while keeping exam-relevant fields.

    out: list[dict[str, Any]] = []
    for i, h in enumerate(hits):
        d = {
            "rank": i + 1,
            "chunk_id": h.get("chunk_id"),
            "score": h.get("score"),
            "final_score": h.get("final_score"),
            "score_01": h.get("score_01"),
            "metadata": h.get("metadata"),
            "text": str(h.get("text", ""))[:max_text],
        }
        out.append(d)
    return out


def _prompt_meta_for_log(meta: dict[str, Any]) -> dict[str, Any]:
        # Omit duplicate huge string where not needed; logger still gets full session.

    slim = {k: v for k, v in meta.items() if k not in ("api_messages", "prompt_log_string")}
    slim["has_api_messages"] = bool(meta.get("api_messages"))
    slim["prompt_char_len"] = len(meta.get("prompt_log_string", ""))
    return slim


def render_sidebar() -> dict[str, Any]:
        # Collect all configuration from the sidebar.

    st.sidebar.header("Configuration")

    st.sidebar.subheader("Data sources")
    use_csv = st.sidebar.checkbox("Include election CSV", value=True)
    use_pdf = st.sidebar.checkbox("Include budget PDF", value=True)

    st.sidebar.subheader("Retrieval")
    mode = st.sidebar.radio("Retrieval mode", ("vector", "hybrid"), horizontal=True)
    top_k = st.sidebar.slider("top_k", min_value=1, max_value=30, value=5)
    hybrid_alpha = st.sidebar.slider("Hybrid α (vector weight)", 0.0, 1.0, 0.7, 0.05)
    hybrid_beta = st.sidebar.slider("Hybrid β (keyword weight)", 0.0, 1.0, 0.3, 0.05)

    st.sidebar.subheader("Chunking")
    chunk_size = st.sidebar.number_input("chunk_size (chars)", 200, 4000, 900, 50)
    overlap = st.sidebar.number_input("overlap (chars)", 0, 500, 120, 10)

    st.sidebar.subheader("Prompting")
    style_key = st.sidebar.selectbox(
        "Prompt style",
        options=[s.value for s in PromptStyle],
        format_func=lambda x: x,
    )
    show_raw_prompt = st.sidebar.checkbox("Show raw prompt in UI", value=True)
    ctx_chars = st.sidebar.number_input("Max context chars (prompt builder)", 1000, 50000, 12000, 500)

    st.sidebar.subheader("Index / cache")
    rebuild = st.sidebar.checkbox("Rebuild processed artifacts & FAISS", value=False)

    st.sidebar.subheader("Embeddings model")
    embedder_model = st.sidebar.text_input(
        "sentence-transformers model id",
        value=DEFAULT_MODEL_NAME,
        help="Changing this requires rebuilding the index (different dimension).",
    )

    st.sidebar.subheader("Generation (OpenAI-compatible)")
    temperature = st.sidebar.slider("temperature", 0.0, 1.5, 0.2, 0.05)
    max_tokens = st.sidebar.number_input("max_tokens (0 = omit cap)", 0, 4096, 512, 64)
    model_override = st.sidebar.text_input(
        "Model override (optional)",
        value=os.environ.get("OPENAI_MODEL", ""),
        help="Leave empty to use OPENAI_MODEL env or default gpt-4o-mini.",
    )

    return {
        "use_csv": use_csv,
        "use_pdf": use_pdf,
        "mode": mode,
        "top_k": int(top_k),
        "hybrid_alpha": float(hybrid_alpha),
        "hybrid_beta": float(hybrid_beta),
        "chunk_size": int(chunk_size),
        "overlap": int(overlap),
        "prompt_style": PromptStyle(style_key),
        "show_raw_prompt": show_raw_prompt,
        "ctx_chars": int(ctx_chars),
        "rebuild": rebuild,
        "embedder_model": embedder_model.strip() or DEFAULT_MODEL_NAME,
        "temperature": float(temperature),
        "max_tokens": int(max_tokens),
        "model_override": model_override.strip(),
    }


def main() -> None:
    _init_session_state()
    st.set_page_config(
        page_title="Introduction to Ai — Academic City",
        page_icon="📚",
        layout="wide",
    )

    cfg = render_sidebar()

    st.title("Academic City - Introduction to Ai")
    st.markdown(
        """
This app runs a **fully manual** RAG stack: ingest → chunk → embed →
FAISS → retrieve → prompt → OpenAI chat. Use the sidebar to change sources,
retrieval mode, and generation settings. Each query logs a JSON session for logging and reproducibility.
        """.strip()
    )

    st.subheader("1. Corpus & vector index")
    build_btn = st.button("Build / load index", type="primary")

    if build_btn:
        _set_stage("load", "running", "")
        with st.spinner("Preparing documents and FAISS index…"):
            index, chunk_dicts, msgs = build_or_load_index(
                rebuild=cfg["rebuild"],
                use_csv=cfg["use_csv"],
                use_pdf=cfg["use_pdf"],
                chunk_size=cfg["chunk_size"],
                overlap=cfg["overlap"],
                embedder_model=cfg["embedder_model"],
            )
        for m in msgs:
            if "failed" in m.lower() or "not found" in m.lower() or "cannot" in m.lower():
                st.warning(m)
            else:
                st.info(m)

        if index is None:
            _set_stage("load", "error", "index missing")
            st.error("Index is not available. Fix errors above, select at least one data source, then rebuild.")
            st.session_state["live_index"] = None
            st.session_state["live_chunks"] = []
        else:
            _set_stage("load", "ok", "")
            _set_stage("chunk", "ok", "")
            _set_stage("embed", "ok", "")
            _set_stage("index", "ok", "")
            st.session_state["live_index"] = index
            st.session_state["live_chunks"] = chunk_dicts
            st.success(f"Index ready: **{index.ntotal}** chunks.")

    # Lazy load index if user skips button but files exist
    if st.session_state.get("live_index") is None:
        manifest_path = FAISS_DIR / MANIFEST_FILENAME
        if manifest_path.is_file() and not cfg["rebuild"]:
            try:
                bust = int(st.session_state.get("index_cache_bust", 0))
                idx = cached_faiss_index(str(FAISS_DIR.resolve()), bust)
                mtime = manifest_path.stat().st_mtime
                ch = load_chunks_from_manifest(str(manifest_path), mtime)
                st.session_state["live_index"] = idx
                st.session_state["live_chunks"] = ch
                st.info(f"Auto-loaded existing index (**{idx.ntotal}** chunks).")
            except Exception:
                st.session_state["live_index"] = None
                st.session_state["live_chunks"] = []

    st.subheader("2. Query")
    query = st.text_input("Your question", placeholder="e.g. What happened in Greater Accra?", key="qbox")
    run = st.button("Run RAG query", type="primary")

    if not run:
        st.stop()

    if cfg["rebuild"]:
        with st.spinner("Rebuild flag is on — regenerating chunks, embeddings, and FAISS…"):
            idx2, ch2, rebuild_msgs = build_or_load_index(
                rebuild=True,
                use_csv=cfg["use_csv"],
                use_pdf=cfg["use_pdf"],
                chunk_size=cfg["chunk_size"],
                overlap=cfg["overlap"],
                embedder_model=cfg["embedder_model"],
            )
        for m in rebuild_msgs:
            if any(x in m.lower() for x in ("fail", "error", "no chunks")):
                st.warning(m)
            else:
                st.info(m)
        if idx2 is not None:
            st.session_state["live_index"] = idx2
            st.session_state["live_chunks"] = ch2

    index_obj: FaissChunkIndex | None = st.session_state.get("live_index")
    chunk_dicts: list[dict[str, Any]] = list(st.session_state.get("live_chunks") or [])

    errors: list[str] = []
    llm_resp: LLMResponse | None = None
    prompt_log = ""
    prompt_meta: dict[str, Any] = {}
    hits: list[dict[str, Any]] = []
    retrieval_mode_used = cfg["mode"]

    if index_obj is None or not chunk_dicts:
        _set_stage("retrieve", "error", "no index")
        st.error("Build or load an index first (section 1).")
        errors.append("Index not built.")
    else:
        _set_stage("retrieve", "running", "")
        try:
            embedder = cached_embedder(cfg["embedder_model"])
            # Reload index through cache so bust/embedder stay consistent after rebuild
            bust = int(st.session_state.get("index_cache_bust", 0))
            index_obj = cached_faiss_index(str(FAISS_DIR.resolve()), bust)
            manifest_path = FAISS_DIR / MANIFEST_FILENAME
            mtime = manifest_path.stat().st_mtime if manifest_path.is_file() else 0.0
            chunk_dicts = load_chunks_from_manifest(str(manifest_path), mtime)

            hits, retrieval_mode_used = run_retrieval(
                mode=cfg["mode"],
                embedder=embedder,
                index=index_obj,
                all_chunk_dicts=chunk_dicts,
                query=query.strip(),
                top_k=cfg["top_k"],
                hybrid_alpha=cfg["hybrid_alpha"],
                hybrid_beta=cfg["hybrid_beta"],
            )
            _set_stage("retrieve", "ok", f"{len(hits)} hits")
        except ValueError as exc:
            _set_stage("retrieve", "error", str(exc))
            msg = str(exc)
            if "dim" in msg.lower() or "embedding" in msg.lower():
                msg += " — Rebuild the index after changing the embedding model."
            errors.append(f"Retrieval: {msg}")
            st.error(msg)
        except Exception as exc:  # noqa: BLE001
            _set_stage("retrieve", "error", str(exc))
            errors.append(f"Retrieval: {exc}")
            st.error(traceback.format_exc())

    if not hits and index_obj is not None:
        st.warning("No retrieval hits returned — the model will see empty context.")
        _set_stage("retrieve", "skipped", "zero hits")

    _set_stage("prompt", "running", "")
    try:
        ctx_cfg = ContextAssemblyConfig(max_context_chars=cfg["ctx_chars"])
        prompt_log, prompt_meta = build_rag_prompt(
            query.strip(),
            hits,
            style=cfg["prompt_style"],
            context_config=ctx_cfg,
        )
        _set_stage("prompt", "ok", "")
    except Exception as exc:  # noqa: BLE001
        _set_stage("prompt", "error", str(exc))
        errors.append(f"Prompt: {exc}")
        st.error(traceback.format_exc())

    _set_stage("generate", "running", "")
    api_msgs = list(prompt_meta.get("api_messages") or [])
    llm_resp, gen_err = run_generation(
        api_messages=api_msgs,
        temperature=cfg["temperature"],
        max_tokens=cfg["max_tokens"],
        model_override=cfg["model_override"],
    )
    if gen_err:
        _set_stage("generate", "skipped", gen_err)
        errors.append(gen_err)
        st.warning(gen_err)
    elif llm_resp and llm_resp.error:
        _set_stage("generate", "error", llm_resp.error)
        errors.append(llm_resp.error)
        st.error(llm_resp.error)
    else:
        _set_stage("generate", "ok", "")

    # --- Logging --------------------------------------------------------------
    logger = RagJsonLogger()
    session_id = logger.new_session_id("ui")
    sidebar_log: dict[str, Any] = {}
    for k, v in cfg.items():
        sidebar_log[k] = v.value if isinstance(v, PromptStyle) else v

    log_paths = logger.log_full_session(
        session_id,
        {
            "query": query.strip(),
            "sidebar_config": sidebar_log,
            "retrieval_mode": retrieval_mode_used,
            "prompt_style": cfg["prompt_style"].value,
            "retrieval_hits": _hits_for_log(hits),
            "prompt_metadata": _prompt_meta_for_log(prompt_meta),
            "prompt_log_string": prompt_log,
            "llm_response": llm_resp.to_log_dict() if llm_resp else None,
            "errors": errors,
        },
        write_snapshot=True,
    )
    st.session_state["last_session_paths"] = log_paths

    # --- Results UI (single page: query above, chunks then answer) ------------
    st.divider()
    st.subheader("Retrieved chunks")
    if not hits:
        st.caption("No chunks to display.")
    else:
        for i, h in enumerate(hits, start=1):
            render_hit(h, i)

    st.divider()
    st.subheader("Final response")
    if llm_resp and not llm_resp.error and llm_resp.output_text.strip():
        st.markdown(llm_resp.output_text)
    elif llm_resp and llm_resp.error:
        st.error(f"Generation error: {llm_resp.error}")
    else:
        st.info("No model output (generation disabled or failed). Retrieved chunks are shown above.")


if __name__ == "__main__":
    main()
