from __future__ import annotations

import argparse
from pathlib import Path

from prototype_core import (
    create_or_reuse_vector_store,
    ensure_dirs,
    get_client,
    list_local_supported_files,
    upload_paths_to_vector_store,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Bulk upload UnVRap papers to an OpenAI vector store."
    )
    parser.add_argument(
        "--folder",
        required=True,
        help="Folder containing PDF/DOCX/TXT/MD/RTF files.",
    )
    parser.add_argument(
        "--vector-store-name",
        default="unvrap-vret-kb",
        help="Name for a new vector store.",
    )
    parser.add_argument(
        "--vector-store-id",
        default="",
        help="Reuse an existing vector store instead of creating a new one.",
    )
    parser.add_argument(
        "--project-name",
        default="UnVRap",
        help="Metadata value stored on uploaded files.",
    )
    parser.add_argument(
        "--topic-name",
        default="VRET",
        help="Metadata value stored on uploaded files.",
    )
    args = parser.parse_args()

    ensure_dirs()
    client = get_client()

    vector_store_id = create_or_reuse_vector_store(
        client=client,
        vector_store_name=args.vector_store_name,
        existing_vector_store_id=args.vector_store_id or None,
    )

    folder = Path(args.folder)
    paths = list_local_supported_files(folder)
    if not paths:
        raise SystemExit(f"No supported files found in {folder}")

    result = upload_paths_to_vector_store(
        client=client,
        paths=paths,
        vector_store_id=vector_store_id,
        project_name=args.project_name,
        topic_name=args.topic_name,
    )

    print("Vector store ID:", vector_store_id)
    print("Uploaded count:", result["uploaded_count"])
    print("Skipped:", result["skipped"])
    print("Batch status:", result["batch_status"])
    if result.get("batch_file_counts") is not None:
        print("Batch file counts:", result["batch_file_counts"])


if __name__ == "__main__":
    main()
