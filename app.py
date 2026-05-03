from __future__ import annotations

import json

import pandas as pd
import streamlit as st

from prototype_core import (
    BRIEF_FILE,
    DEFAULT_MODEL,
    MANIFEST_FILE,
    STATE_FILE,
    batch_generate_all_summaries,
    batch_generate_all_summaries_local,
    create_or_reuse_vector_store,
    ensure_dirs,
    generate_literature_brief,
    generate_literature_brief_local,
    generate_single_paper_summary,
    generate_single_paper_summary_local,
    get_all_doc_records,
    humanize_literature_brief_package,
    humanize_summary_package,
    get_client,
    get_doc_record,
    get_evidence_chunks,
    get_evidence_chunks_local,
    load_manifest,
    load_project_state,
    load_summary_packages,
    register_local_paths,
    save_uploaded_streams,
    upload_paths_to_vector_store,
)

st.set_page_config(page_title="UnVRap VRET Summarizer", layout="wide")
ensure_dirs()

st.title("UnVRap VRET Research Summarizer")
st.caption(
    "Student prototype: save VRET papers locally, then summarize them either in Local no-cost mode or in Cloud mode if you later get API credits."
)

with st.sidebar:
    st.header("Project setup")
    processing_mode = st.selectbox("Processing mode", ["Local no-cost", "Cloud (OpenAI)"], index=0)
    model_name = st.text_input("Model name", value=DEFAULT_MODEL)
    state = load_project_state()
    vector_store_name = st.text_input(
        "Vector store name",
        value=state.get("vector_store_name") or "unvrap-vret-kb",
    )
    existing_vector_store_id = st.text_input(
        "Existing vector store ID (optional)",
        value=state.get("vector_store_id") or "",
        help="Leave empty to create a new vector store. Paste an existing ID if you already uploaded files before.",
    )

    if st.button("Create / reuse vector store", use_container_width=True):
        if processing_mode == "Local no-cost":
            st.info("Local no-cost mode does not need an OpenAI vector store. You can skip this step.")
        else:
            try:
                client = get_client()
                vector_store_id = create_or_reuse_vector_store(
                    client=client,
                    vector_store_name=vector_store_name,
                    existing_vector_store_id=existing_vector_store_id or None,
                )
                st.session_state["vector_store_id"] = vector_store_id
                st.success(f"Vector store ready: {vector_store_id}")
            except Exception as exc:
                st.error(str(exc))

    current_state = load_project_state()
    st.markdown("---")
    st.write("**Current vector store ID**")
    st.code(current_state.get("vector_store_id", "Not set yet"))
    st.write("**State file**")
    st.code(str(STATE_FILE))
    st.write("**Manifest file**")
    st.code(str(MANIFEST_FILE))
    st.write("**Literature brief file**")
    st.code(str(BRIEF_FILE))

vector_store_id = load_project_state().get("vector_store_id", "")

upload_tab, single_tab, batch_tab, brief_tab = st.tabs(
    [
        "1. Upload & index",
        "2. Single paper summary",
        "3. Batch summarize all papers",
        "4. Literature brief",
    ]
)

with upload_tab:
    st.subheader("Upload all documents shared by UnVRap")
    st.write(
        "Use this first. In Local no-cost mode, only the local save step is required. In Cloud mode, you can also upload into the vector store."
    )

    uploaded_files = st.file_uploader(
        "Choose documents",
        type=["pdf", "docx", "txt", "md", "rtf"],
        accept_multiple_files=True,
    )

    col1, col2 = st.columns([1, 1])
    with col1:
        save_only = st.button("Save uploaded files locally", use_container_width=True)
    with col2:
        upload_and_index = st.button("Upload + index in OpenAI", use_container_width=True)

    if save_only:
        if not uploaded_files:
            st.warning("Choose at least one file first.")
        else:
            saved_paths = save_uploaded_streams(uploaded_files)
            registration = register_local_paths(saved_paths)
            st.success(f"Saved {len(saved_paths)} files into the local data/uploads folder and registered {registration['registered_count']} files in the manifest.")
            if registration.get("skipped"):
                st.info(f"Skipped already-registered files: {', '.join(registration['skipped'])}")
            st.write([str(path) for path in saved_paths])

    if upload_and_index:
        if processing_mode == "Local no-cost":
            st.info("Local no-cost mode does not need cloud indexing. Save files locally, then summarize from the next tab.")
        elif not vector_store_id:
            st.error("Create or reuse a vector store first from the sidebar.")
        elif not uploaded_files:
            st.warning("Choose at least one file first.")
        else:
            try:
                client = get_client()
                saved_paths = save_uploaded_streams(uploaded_files)
                result = upload_paths_to_vector_store(
                    client=client,
                    paths=saved_paths,
                    vector_store_id=vector_store_id,
                )
                st.success(
                    f"Upload finished. Uploaded {result['uploaded_count']} new files. Batch status: {result['batch_status']}"
                )
                if result.get("skipped"):
                    st.info(f"Skipped files: {', '.join(result['skipped'])}")
                if result.get("batch_file_counts"):
                    st.json(result["batch_file_counts"], expanded=False)
            except Exception as exc:
                st.error(str(exc))

    manifest = load_manifest()
    if manifest.get("files"):
        st.markdown("### Uploaded document manifest")
        df = pd.DataFrame(manifest["files"])
        st.dataframe(df, use_container_width=True, hide_index=True)

with single_tab:
    st.subheader("Generate one paper summary")
    doc_records = get_all_doc_records()
    if not doc_records:
        st.info("No uploaded documents found yet. Save or upload papers first.")
    else:
        labels = {
            record["doc_key"]: f"{record['filename']}  |  {record.get('summary_status', 'not_started')}"
            for record in doc_records
        }
        selected_doc_key = st.selectbox(
            "Select a paper",
            options=list(labels.keys()),
            format_func=lambda key: labels[key],
        )

        col1, col2 = st.columns([1, 1])
        with col1:
            run_one = st.button("Generate selected paper summary", use_container_width=True)
        with col2:
            show_evidence = st.button("Show evidence chunks", use_container_width=True)

        if run_one:
            try:
                doc_record = get_doc_record(selected_doc_key)
                if processing_mode == "Local no-cost":
                    package = generate_single_paper_summary_local(doc_record=doc_record)
                else:
                    client = get_client()
                    package = generate_single_paper_summary(
                        client=client,
                        vector_store_id=vector_store_id,
                        doc_record=doc_record,
                        model_name=model_name,
                    )
                st.success("Summary generated and saved.")
                human_summary = humanize_summary_package(package)
                st.markdown(human_summary)

                download_col1, download_col2 = st.columns([1, 1])
                with download_col1:
                    st.download_button(
                        "Download human-readable summary (.md)",
                        data=human_summary,
                        file_name=f"summary_{doc_record['filename']}.md",
                        mime="text/markdown",
                        use_container_width=True,
                    )
                with download_col2:
                    st.download_button(
                        "Download raw JSON",
                        data=json.dumps(package, ensure_ascii=False, indent=2),
                        file_name=f"summary_{doc_record['filename']}.json",
                        mime="application/json",
                        use_container_width=True,
                    )

                with st.expander("View raw structured data (JSON)"):
                    st.json(package["summary"], expanded=False)
            except Exception as exc:
                st.error(str(exc))

        if show_evidence:
            try:
                doc_record = get_doc_record(selected_doc_key)
                if processing_mode == "Local no-cost":
                    chunks = get_evidence_chunks_local(doc_record=doc_record, max_num_results=5)
                else:
                    client = get_client()
                    chunks = get_evidence_chunks(
                        client=client,
                        vector_store_id=vector_store_id,
                        doc_key=selected_doc_key,
                        max_num_results=5,
                    )
                if not chunks:
                    st.warning("No evidence chunks were returned.")
                for idx, chunk in enumerate(chunks, start=1):
                    st.markdown(f"**Chunk {idx}** | similarity score: {chunk.get('score')}")
                    st.text_area(
                        label=f"Evidence {idx}",
                        value=chunk.get("text", ""),
                        height=180,
                    )
            except Exception as exc:
                st.error(str(exc))

with batch_tab:
    st.subheader("Generate summaries for every uploaded paper")
    st.write(
        "Run this after one or two sample papers look sensible. It saves one structured JSON summary and one human-readable Markdown summary per paper."
    )
    if st.button("Batch summarize all uploaded papers", use_container_width=True):
        try:
            if processing_mode == "Local no-cost":
                packages = batch_generate_all_summaries_local()
            else:
                client = get_client()
                packages = batch_generate_all_summaries(
                    client=client,
                    vector_store_id=vector_store_id,
                    model_name=model_name,
                )
            st.success(f"Generated {len(packages)} paper summaries.")
        except Exception as exc:
            st.error(str(exc))

    summary_packages = load_summary_packages()
    if summary_packages:
        st.markdown("### Saved summary files")
        summary_rows = []
        for package in summary_packages:
            summary_rows.append(
                {
                    "filename": package.get("filename"),
                    "generated_at": package.get("generated_at"),
                    "model_name": package.get("model_name"),
                    "paper_title": package.get("summary", {}).get("paper_title"),
                    "phobia_target": package.get("summary", {}).get("phobia_target"),
                    "confidence": package.get("summary", {}).get("confidence"),
                }
            )
        st.dataframe(pd.DataFrame(summary_rows), use_container_width=True, hide_index=True)

with brief_tab:
    st.subheader("Create a literature brief from saved summaries")
    st.write(
        "This stage is cheaper and more stable than summarizing all raw papers at once because it synthesizes the saved structured summaries."
    )
    if st.button("Generate literature brief", use_container_width=True):
        try:
            summary_packages = load_summary_packages()
            if processing_mode == "Local no-cost":
                package = generate_literature_brief_local(summary_packages=summary_packages)
            else:
                client = get_client()
                package = generate_literature_brief(
                    client=client,
                    summary_packages=summary_packages,
                    model_name=model_name,
                )
            st.success("Literature brief generated and saved.")
            human_brief = humanize_literature_brief_package(package)
            st.markdown(human_brief)

            download_col1, download_col2 = st.columns([1, 1])
            with download_col1:
                st.download_button(
                    "Download human-readable literature brief (.md)",
                    data=human_brief,
                    file_name="unvrap_vret_literature_brief.md",
                    mime="text/markdown",
                    use_container_width=True,
                )
            with download_col2:
                st.download_button(
                    "Download raw JSON",
                    data=json.dumps(package, ensure_ascii=False, indent=2),
                    file_name="unvrap_vret_literature_brief.json",
                    mime="application/json",
                    use_container_width=True,
                )

            with st.expander("View raw structured data (JSON)"):
                st.json(package["brief"], expanded=False)
        except Exception as exc:
            st.error(str(exc))

    st.markdown("### Prototype workflow")
    st.code(
        """
1) Choose Local no-cost mode for zero API spend, or Cloud mode if you later have credits
2) Save all UnVRap papers locally
3) In Cloud mode only: create/reuse vector store and upload papers
4) Test one selected paper summary
5) Inspect evidence chunks
6) Batch summarize all papers
7) Generate literature brief from saved summaries
8) Share the human-readable Markdown outputs with UnVRap and use raw JSON only as the internal structured format
9) Review outputs with UnVRap and refine prompts/schema
        """.strip()
    )
