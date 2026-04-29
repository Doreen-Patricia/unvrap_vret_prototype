from __future__ import annotations

import argparse

from prototype_core import (
    DEFAULT_MODEL,
    batch_generate_all_summaries,
    get_client,
    load_project_state,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate one structured summary JSON per uploaded paper."
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help="Model name for summary generation.",
    )
    parser.add_argument(
        "--vector-store-id",
        default="",
        help="Optional explicit vector store ID. If omitted, the saved project state is used.",
    )
    args = parser.parse_args()

    client = get_client()
    vector_store_id = args.vector_store_id or load_project_state().get("vector_store_id", "")
    if not vector_store_id:
        raise SystemExit("No vector store ID found. Upload documents first.")

    packages = batch_generate_all_summaries(
        client=client,
        vector_store_id=vector_store_id,
        model_name=args.model,
    )
    print(f"Generated {len(packages)} summaries.")


if __name__ == "__main__":
    main()
