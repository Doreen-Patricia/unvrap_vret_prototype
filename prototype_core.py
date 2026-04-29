from __future__ import annotations

import json
import os
import re
import uuid
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv
from openai import OpenAI

try:
    import fitz  # PyMuPDF
except ImportError:  # pragma: no cover - optional local fallback dependency
    fitz = None

try:
    from docx import Document
except ImportError:  # pragma: no cover - optional local fallback dependency
    Document = None

BASE_DIR = Path(__file__).resolve().parent
load_dotenv(dotenv_path=BASE_DIR / ".env", override=True)
DATA_DIR = BASE_DIR / "data" / "uploads"
OUTPUTS_DIR = BASE_DIR / "outputs"
SUMMARIES_DIR = OUTPUTS_DIR / "paper_summaries"
STATE_FILE = OUTPUTS_DIR / "project_state.json"
MANIFEST_FILE = OUTPUTS_DIR / "upload_manifest.json"
BRIEF_FILE = OUTPUTS_DIR / "literature_brief.json"

SUPPORTED_SUFFIXES = {".pdf", ".docx", ".txt", ".md", ".rtf"}
DEFAULT_MODEL = os.getenv("MODEL_NAME", "gpt-4.1-mini")

SUMMARY_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "properties": {
        "paper_title": {"type": "string"},
        "publication_year": {"type": "string"},
        "authors": {"type": "string"},
        "paper_type": {"type": "string"},
        "phobia_target": {"type": "string"},
        "study_design": {"type": "string"},
        "participant_details": {"type": "string"},
        "vr_setup_and_exposure_protocol": {"type": "string"},
        "comparison_or_control": {"type": "string"},
        "outcome_measures": {"type": "array", "items": {"type": "string"}},
        "key_findings": {"type": "array", "items": {"type": "string"}},
        "limitations": {"type": "array", "items": {"type": "string"}},
        "relevance_to_unvrap": {"type": "string"},
        "recommended_action_for_unvrap": {"type": "string"},
        "confidence": {"type": "string", "enum": ["high", "medium", "low"]},
        "concise_summary": {"type": "string"},
    },
    "required": [
        "paper_title",
        "publication_year",
        "authors",
        "paper_type",
        "phobia_target",
        "study_design",
        "participant_details",
        "vr_setup_and_exposure_protocol",
        "comparison_or_control",
        "outcome_measures",
        "key_findings",
        "limitations",
        "relevance_to_unvrap",
        "recommended_action_for_unvrap",
        "confidence",
        "concise_summary",
    ],
    "additionalProperties": False,
}

LITERATURE_BRIEF_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "properties": {
        "number_of_papers_considered": {"type": "integer"},
        "recurring_phobias": {"type": "array", "items": {"type": "string"}},
        "recurring_methods": {"type": "array", "items": {"type": "string"}},
        "common_outcomes": {"type": "array", "items": {"type": "string"}},
        "common_limitations": {"type": "array", "items": {"type": "string"}},
        "strongest_signals_for_unvrap": {"type": "array", "items": {"type": "string"}},
        "research_gaps": {"type": "array", "items": {"type": "string"}},
        "implementation_note": {"type": "string"},
        "executive_summary": {"type": "string"},
    },
    "required": [
        "number_of_papers_considered",
        "recurring_phobias",
        "recurring_methods",
        "common_outcomes",
        "common_limitations",
        "strongest_signals_for_unvrap",
        "research_gaps",
        "implementation_note",
        "executive_summary",
    ],
    "additionalProperties": False,
}


def ensure_dirs() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    SUMMARIES_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)


def get_client() -> OpenAI:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError(
            "OPENAI_API_KEY not found. Create a .env file and set OPENAI_API_KEY=your_key"
        )
    return OpenAI(api_key=api_key)


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def slugify(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r"[^a-z0-9]+", "-", text)
    return text.strip("-") or "document"


def save_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def load_json(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_project_state() -> Dict[str, Any]:
    return load_json(STATE_FILE, {"vector_store_id": "", "vector_store_name": ""})


def save_project_state(vector_store_id: str, vector_store_name: str) -> None:
    save_json(
        STATE_FILE,
        {
            "vector_store_id": vector_store_id,
            "vector_store_name": vector_store_name,
            "updated_at": now_iso(),
        },
    )


def load_manifest() -> Dict[str, Any]:
    return load_json(MANIFEST_FILE, {"files": []})


def save_manifest(manifest: Dict[str, Any]) -> None:
    save_json(MANIFEST_FILE, manifest)


def get_existing_filename_set(manifest: Dict[str, Any]) -> set[str]:
    return {item["filename"] for item in manifest.get("files", [])}


def list_local_supported_files(folder: Path) -> List[Path]:
    folder = Path(folder)
    if not folder.exists():
        raise FileNotFoundError(f"Folder not found: {folder}")
    return sorted(
        [p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in SUPPORTED_SUFFIXES],
        key=lambda p: p.name.lower(),
    )


def create_or_reuse_vector_store(
    client: OpenAI,
    vector_store_name: str,
    existing_vector_store_id: Optional[str] = None,
) -> str:
    if existing_vector_store_id:
        save_project_state(existing_vector_store_id, vector_store_name)
        return existing_vector_store_id

    vector_store = client.vector_stores.create(name=vector_store_name)
    save_project_state(vector_store.id, vector_store_name)
    return vector_store.id


def save_uploaded_streams(uploaded_files: List[Any]) -> List[Path]:
    saved_paths: List[Path] = []
    for uploaded_file in uploaded_files:
        output_path = DATA_DIR / uploaded_file.name
        with output_path.open("wb") as f:
            f.write(uploaded_file.getbuffer())
        saved_paths.append(output_path)
    return saved_paths


def register_local_paths(paths: List[Path], project_name: str = "UnVRap", topic_name: str = "VRET") -> Dict[str, Any]:
    ensure_dirs()
    manifest = load_manifest()
    existing_filenames = get_existing_filename_set(manifest)
    new_records: List[Dict[str, Any]] = []
    skipped: List[str] = []

    for path in paths:
        if path.suffix.lower() not in SUPPORTED_SUFFIXES:
            skipped.append(path.name)
            continue
        if path.name in existing_filenames:
            skipped.append(path.name)
            continue
        doc_key = f"{slugify(path.stem)}-{uuid.uuid4().hex[:8]}"
        new_records.append(
            {
                "doc_key": doc_key,
                "filename": path.name,
                "local_path": str(path),
                "file_id": None,
                "vector_store_id": load_project_state().get("vector_store_id", ""),
                "project": project_name,
                "topic": topic_name,
                "uploaded_at": now_iso(),
                "summary_status": "not_started",
            }
        )

    if new_records:
        manifest["files"].extend(new_records)
        manifest["updated_at"] = now_iso()
        save_manifest(manifest)

    return {"registered_count": len(new_records), "skipped": skipped, "records": new_records}


def upload_paths_to_vector_store(
    client: OpenAI,
    paths: List[Path],
    vector_store_id: str,
    project_name: str = "UnVRap",
    topic_name: str = "VRET",
    skip_existing: bool = True,
) -> Dict[str, Any]:
    ensure_dirs()
    manifest = load_manifest()
    existing_filenames = get_existing_filename_set(manifest)

    paths_to_upload: List[Path] = []
    skipped: List[str] = []
    for path in paths:
        if path.suffix.lower() not in SUPPORTED_SUFFIXES:
            skipped.append(path.name)
            continue
        if skip_existing and path.name in existing_filenames:
            skipped.append(path.name)
            continue
        paths_to_upload.append(path)

    if not paths_to_upload:
        return {
            "uploaded_count": 0,
            "skipped": skipped,
            "records": [],
            "batch_status": "nothing_to_upload",
        }

    batch_files: List[Dict[str, Any]] = []
    new_records: List[Dict[str, Any]] = []

    for path in paths_to_upload:
        doc_key = f"{slugify(path.stem)}-{uuid.uuid4().hex[:8]}"
        with path.open("rb") as f:
            openai_file = client.files.create(file=f, purpose="assistants")

        batch_files.append(
            {
                "file_id": openai_file.id,
                "attributes": {
                    "doc_key": doc_key,
                    "filename": path.name,
                    "project": project_name,
                    "topic": topic_name,
                    "source": "uploaded",
                },
            }
        )
        new_records.append(
            {
                "doc_key": doc_key,
                "filename": path.name,
                "local_path": str(path),
                "file_id": openai_file.id,
                "vector_store_id": vector_store_id,
                "uploaded_at": now_iso(),
                "summary_status": "not_started",
            }
        )

    batch = client.vector_stores.file_batches.create_and_poll(
        vector_store_id=vector_store_id,
        files=batch_files,
    )

    manifest["files"].extend(new_records)
    manifest["updated_at"] = now_iso()
    manifest["vector_store_id"] = vector_store_id
    save_manifest(manifest)

    return {
        "uploaded_count": len(new_records),
        "skipped": skipped,
        "records": new_records,
        "batch_status": getattr(batch, "status", "unknown"),
        "batch_file_counts": getattr(batch, "file_counts", None),
    }


def get_doc_record(doc_key: str) -> Dict[str, Any]:
    manifest = load_manifest()
    for item in manifest.get("files", []):
        if item["doc_key"] == doc_key:
            return item
    raise KeyError(f"Document not found: {doc_key}")


def get_all_doc_records() -> List[Dict[str, Any]]:
    return load_manifest().get("files", [])


def build_summary_prompt(filename: str) -> List[Dict[str, str]]:
    system_prompt = (
        "You are an evidence-grounded research assistant working for UnVRap, a B2B VR SaaS company. "
        "Your job is to summarize academic papers about Virtual Reality Exposure Therapy (VRET) for phobias. "
        "Use only the retrieved content from the selected paper. "
        "If a detail is missing, write 'Not reported'. "
        "Do not invent facts. Do not overstate clinical effectiveness. "
        "Keep findings precise and useful for product and research teams."
    )

    user_prompt = (
        f"Create a structured summary for the paper with filename '{filename}'. "
        "Focus on: paper metadata, phobia target, study design, participants, VR setup, exposure protocol, "
        "comparison/control, outcome measures, key findings, limitations, relevance to UnVRap, and a recommended next action. "
        "If the retrieved evidence is weak or incomplete, lower the confidence level."
    )

    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]


def generate_single_paper_summary(
    client: OpenAI,
    vector_store_id: str,
    doc_record: Dict[str, Any],
    model_name: str = DEFAULT_MODEL,
) -> Dict[str, Any]:
    response = client.responses.create(
        model=model_name,
        input=build_summary_prompt(doc_record["filename"]),
        tools=[
            {
                "type": "file_search",
                "vector_store_ids": [vector_store_id],
                "max_num_results": 12,
                "filters": {
                    "type": "eq",
                    "key": "doc_key",
                    "value": doc_record["doc_key"],
                },
            }
        ],
        include=["file_search_call.results"],
        text={
            "format": {
                "type": "json_schema",
                "name": "vret_paper_summary",
                "schema": SUMMARY_SCHEMA,
                "strict": True,
            }
        },
    )

    if not getattr(response, "output_text", None):
        raise RuntimeError("No structured response received from the model.")

    summary = json.loads(response.output_text)
    package = {
        "doc_key": doc_record["doc_key"],
        "filename": doc_record["filename"],
        "vector_store_id": vector_store_id,
        "model_name": model_name,
        "generated_at": now_iso(),
        "summary": summary,
    }

    output_path = SUMMARIES_DIR / f"{doc_record['doc_key']}.json"
    save_json(output_path, package)
    update_summary_status(doc_record["doc_key"], status="completed")
    return package


def update_summary_status(doc_key: str, status: str) -> None:
    manifest = load_manifest()
    updated = False
    for item in manifest.get("files", []):
        if item["doc_key"] == doc_key:
            item["summary_status"] = status
            item["summary_updated_at"] = now_iso()
            updated = True
            break
    if updated:
        save_manifest(manifest)


def load_summary_packages() -> List[Dict[str, Any]]:
    if not SUMMARIES_DIR.exists():
        return []
    packages: List[Dict[str, Any]] = []
    for path in sorted(SUMMARIES_DIR.glob("*.json")):
        packages.append(load_json(path, {}))
    return packages


def generate_literature_brief(
    client: OpenAI,
    summary_packages: List[Dict[str, Any]],
    model_name: str = DEFAULT_MODEL,
) -> Dict[str, Any]:
    if not summary_packages:
        raise ValueError("No paper summaries found. Generate individual summaries first.")

    compact_payload = [
        {
            "filename": package.get("filename", ""),
            **package.get("summary", {}),
        }
        for package in summary_packages
    ]

    response = client.responses.create(
        model=model_name,
        input=[
            {
                "role": "system",
                "content": (
                    "You are preparing a literature brief for UnVRap. "
                    "Use only the structured paper summaries provided. "
                    "Identify repeated patterns, practical implications, and gaps. "
                    "Be specific and avoid unsupported claims."
                ),
            },
            {
                "role": "user",
                "content": (
                    "Create a literature brief across the provided VRET paper summaries. "
                    f"Here are the summaries in JSON:\n\n{json.dumps(compact_payload, ensure_ascii=False, indent=2)}"
                ),
            },
        ],
        text={
            "format": {
                "type": "json_schema",
                "name": "vret_literature_brief",
                "schema": LITERATURE_BRIEF_SCHEMA,
                "strict": True,
            }
        },
    )

    if not getattr(response, "output_text", None):
        raise RuntimeError("No literature brief was returned by the model.")

    brief = json.loads(response.output_text)
    package = {
        "generated_at": now_iso(),
        "model_name": model_name,
        "brief": brief,
    }
    save_json(BRIEF_FILE, package)
    return package


def get_evidence_chunks(
    client: OpenAI,
    vector_store_id: str,
    doc_key: str,
    max_num_results: int = 5,
) -> List[Dict[str, Any]]:
    query = (
        "study design, participants, VR setup, exposure protocol, outcome measures, "
        "key findings, limitations, and conclusions"
    )
    result_page = client.vector_stores.search(
        vector_store_id=vector_store_id,
        query=query,
        filters={"type": "eq", "key": "doc_key", "value": doc_key},
        max_num_results=max_num_results,
    )

    chunks: List[Dict[str, Any]] = []
    for item in getattr(result_page, "data", []):
        text_parts: List[str] = []
        for content_item in getattr(item, "content", []) or []:
            text_value = getattr(content_item, "text", None)
            if text_value:
                text_parts.append(text_value)
            elif isinstance(content_item, dict) and content_item.get("text"):
                text_parts.append(content_item["text"])
        chunks.append(
            {
                "filename": getattr(item, "filename", None) or getattr(item, "file_name", None),
                "file_id": getattr(item, "file_id", None),
                "score": getattr(item, "score", None),
                "text": "\n\n".join(text_parts).strip(),
            }
        )
    return chunks




STOPWORDS = {
    "the", "and", "for", "that", "with", "this", "from", "are", "was", "were", "have", "has", "had",
    "into", "their", "there", "about", "which", "while", "than", "also", "been", "being", "between",
    "using", "used", "use", "via", "into", "over", "under", "among", "such", "more", "most", "less",
    "than", "then", "they", "them", "these", "those", "when", "where", "what", "who", "whom", "whose",
    "your", "our", "his", "her", "its", "not", "but", "can", "could", "should", "would", "may", "might",
    "each", "other", "some", "many", "few", "very", "within", "without", "after", "before", "during",
    "into", "onto", "upon", "all", "any", "both", "either", "neither", "because", "however", "therefore",
    "study", "paper", "article", "research", "result", "results", "method", "methods"
}

PHOBIA_PATTERNS = [
    (r"arachnophobia|spider phobia|fear of spiders", "Arachnophobia / spider phobia"),
    (r"acrophobia|fear of heights|height phobia", "Acrophobia / fear of heights"),
    (r"aviophobia|fear of flying|flying phobia", "Aviophobia / fear of flying"),
    (r"social phobia|social anxiety|public speaking anxiety", "Social anxiety / social phobia"),
    (r"claustrophobia|fear of enclosed spaces", "Claustrophobia"),
    (r"agoraphobia", "Agoraphobia"),
    (r"dental phobia|fear of dentists|dental anxiety", "Dental phobia / dental anxiety"),
    (r"fear of animals|animal phobia", "Animal phobia"),
]


def normalize_text(text: str) -> str:
    text = text.replace(" ", " ")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def extract_text_from_path(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        if fitz is None:
            raise RuntimeError("PyMuPDF is required for local PDF summarization. Install it with: pip install pymupdf")
        with fitz.open(path) as doc:
            return "\n".join(page.get_text("text") for page in doc)
    if suffix == ".docx":
        if Document is None:
            raise RuntimeError("python-docx is required for local DOCX summarization. Install it with: pip install python-docx")
        doc = Document(path)
        return "\n".join(p.text for p in doc.paragraphs)
    return path.read_text(encoding="utf-8", errors="ignore")


def split_sentences(text: str) -> List[str]:
    cleaned = re.sub(r"\s+", " ", text).strip()
    if not cleaned:
        return []
    parts = re.split(r"(?<=[.!?])\s+(?=[A-Z0-9])", cleaned)
    return [p.strip() for p in parts if len(p.strip()) > 40]


def score_sentences(sentences: List[str]) -> List[Dict[str, Any]]:
    tokens = re.findall(r"[A-Za-z][A-Za-z-]{2,}", " ".join(sentences).lower())
    freqs = Counter(tok for tok in tokens if tok not in STOPWORDS)
    boosted_terms = {"participant", "sample", "virtual", "reality", "exposure", "result", "finding", "significant", "control", "limitation", "conclusion"}
    scored: List[Dict[str, Any]] = []
    for idx, sentence in enumerate(sentences):
        words = re.findall(r"[A-Za-z][A-Za-z-]{2,}", sentence.lower())
        if not words:
            continue
        score = sum(freqs.get(w, 0) for w in words if w not in STOPWORDS) / max(len(words), 1)
        score += 0.5 * sum(1 for w in words if w in boosted_terms)
        scored.append({"index": idx, "sentence": sentence, "score": round(score, 3)})
    return scored


def top_sentences(text: str, count: int = 6) -> List[str]:
    sentences = split_sentences(text)
    ranked = score_sentences(sentences)
    top = sorted(ranked, key=lambda x: x["score"], reverse=True)[:count]
    top_sorted = sorted(top, key=lambda x: x["index"])
    return [item["sentence"] for item in top_sorted]


def find_first_match(text: str, patterns: List[str]) -> str:
    sentences = split_sentences(text)
    for sentence in sentences:
        lower = sentence.lower()
        if any(p in lower for p in patterns):
            return sentence
    return "Not reported"


def find_matches(text: str, patterns: List[str], limit: int = 3) -> List[str]:
    results: List[str] = []
    sentences = split_sentences(text)
    for sentence in sentences:
        lower = sentence.lower()
        if any(p in lower for p in patterns):
            results.append(sentence)
        if len(results) >= limit:
            break
    return results


def guess_title(text: str, filename: str) -> str:
    raw_lines = [line.strip() for line in text.splitlines() if line.strip()]
    for line in raw_lines[:10]:
        if 6 <= len(line.split()) <= 30 and not line.lower().startswith(("abstract", "introduction", "keywords")):
            return line
    return Path(filename).stem.replace("_", " ").replace("-", " ").title()


def guess_year(text: str, filename: str) -> str:
    year_match = re.search(r"\b(19|20)\d{2}\b", filename)
    if year_match:
        return year_match.group(0)
    body_match = re.search(r"\b(19|20)\d{2}\b", text)
    return body_match.group(0) if body_match else "Not reported"


def guess_phobia(text: str) -> str:
    lower = text.lower()
    for pattern, label in PHOBIA_PATTERNS:
        if re.search(pattern, lower):
            return label
    return "General phobia / not clearly specified"


def guess_study_design(text: str) -> str:
    lower = text.lower()
    patterns = [
        ("randomized controlled trial", "Randomized controlled trial"),
        ("controlled trial", "Controlled trial"),
        ("pilot study", "Pilot study"),
        ("feasibility study", "Feasibility study"),
        ("case study", "Case study"),
        ("systematic review", "Systematic review"),
        ("meta-analysis", "Meta-analysis"),
        ("review", "Literature review"),
        ("pretest", "Pre/post intervention study"),
    ]
    for pattern, label in patterns:
        if pattern in lower:
            return label
    return "Empirical VRET study"


def guess_paper_type(study_design: str) -> str:
    if "review" in study_design.lower() or "meta-analysis" in study_design.lower():
        return "Review paper"
    if "case study" in study_design.lower():
        return "Case study"
    return "Primary empirical study"


def concise_list_from_sentences(sentences: List[str], fallback: str = "Not reported") -> List[str]:
    cleaned = []
    for sentence in sentences:
        s = normalize_text(sentence)
        if s and s not in cleaned:
            cleaned.append(s)
    return cleaned if cleaned else [fallback]


def generate_single_paper_summary_local(doc_record: Dict[str, Any]) -> Dict[str, Any]:
    local_path = Path(doc_record.get("local_path", ""))
    if not local_path.exists():
        raise FileNotFoundError(f"Local file not found: {local_path}")

    raw_text = extract_text_from_path(local_path)
    text = normalize_text(raw_text)
    if len(text) < 200:
        raise RuntimeError("The document text is too short to summarize locally.")

    summary_sentences = top_sentences(text, count=6)
    finding_sentences = find_matches(text, ["result", "finding", "significant", "improved", "reduction", "decrease", "increase", "effective"], limit=4)
    limitation_sentences = find_matches(text, ["limitation", "small sample", "future research", "lack of control", "follow-up", "generalizability", "generalisability"], limit=3)

    study_design = guess_study_design(text)
    phobia_target = guess_phobia(text)
    participant_details = find_first_match(text, ["participants", "patients", "subjects", "sample", "n =", "n="])
    vr_setup = find_first_match(text, ["virtual reality", "head-mounted", "hmd", "exposure", "session", "simulated"])
    control = find_first_match(text, ["control", "comparison", "waitlist", "treatment as usual", "baseline", "cognitive behavioral"])
    outcome_lines = concise_list_from_sentences(find_matches(text, ["outcome", "measure", "questionnaire", "scale", "inventory", "assessment"], limit=4))

    summary = {
        "paper_title": guess_title(raw_text, doc_record["filename"]),
        "publication_year": guess_year(raw_text, doc_record["filename"]),
        "authors": "Not reliably extracted in local no-cost mode",
        "paper_type": guess_paper_type(study_design),
        "phobia_target": phobia_target,
        "study_design": study_design,
        "participant_details": participant_details,
        "vr_setup_and_exposure_protocol": vr_setup,
        "comparison_or_control": control,
        "outcome_measures": outcome_lines,
        "key_findings": concise_list_from_sentences(finding_sentences or summary_sentences[:3]),
        "limitations": concise_list_from_sentences(limitation_sentences, fallback="Not clearly stated in extracted text"),
        "relevance_to_unvrap": (
            f"This paper appears relevant to UnVRap because it discusses VRET for {phobia_target.lower()} and provides practical details on exposure delivery or outcomes that can inform product design."
            if phobia_target != "General phobia / not clearly specified"
            else "This paper appears relevant because it discusses VRET implementation, outcomes, or design choices that may inform UnVRap's research workflow."
        ),
        "recommended_action_for_unvrap": "Review this paper manually to verify participant details, outcome measures, and effect strength before using it in product or clinical recommendations.",
        "confidence": "medium" if len(summary_sentences) >= 5 else "low",
        "concise_summary": " ".join(summary_sentences[:4]),
    }

    package = {
        "doc_key": doc_record["doc_key"],
        "filename": doc_record["filename"],
        "vector_store_id": doc_record.get("vector_store_id", ""),
        "model_name": "local-extractive-no-cost",
        "generated_at": now_iso(),
        "summary": summary,
    }
    output_path = SUMMARIES_DIR / f"{doc_record['doc_key']}.json"
    save_json(output_path, package)
    update_summary_status(doc_record["doc_key"], status="completed_local")
    return package


def get_evidence_chunks_local(doc_record: Dict[str, Any], max_num_results: int = 5) -> List[Dict[str, Any]]:
    local_path = Path(doc_record.get("local_path", ""))
    if not local_path.exists():
        raise FileNotFoundError(f"Local file not found: {local_path}")
    raw_text = extract_text_from_path(local_path)
    sentences = split_sentences(raw_text)
    ranked = score_sentences(sentences)
    top = sorted(ranked, key=lambda x: x["score"], reverse=True)[:max_num_results]
    top_sorted = sorted(top, key=lambda x: x["index"])
    return [
        {
            "filename": doc_record["filename"],
            "file_id": doc_record.get("file_id"),
            "score": item["score"],
            "text": item["sentence"],
        }
        for item in top_sorted
    ]


def batch_generate_all_summaries_local() -> List[Dict[str, Any]]:
    packages: List[Dict[str, Any]] = []
    for doc_record in get_all_doc_records():
        package = generate_single_paper_summary_local(doc_record)
        packages.append(package)
    return packages


def generate_literature_brief_local(summary_packages: List[Dict[str, Any]]) -> Dict[str, Any]:
    if not summary_packages:
        raise ValueError("No paper summaries found. Generate individual summaries first.")

    phobias = Counter()
    methods = Counter()
    outcomes: List[str] = []
    limitations: List[str] = []
    strong_signals: List[str] = []

    for package in summary_packages:
        summary = package.get("summary", {})
        phobias[summary.get("phobia_target", "Not reported")] += 1
        methods[summary.get("study_design", "Not reported")] += 1
        outcomes.extend(summary.get("key_findings", []))
        limitations.extend(summary.get("limitations", []))
        if summary.get("relevance_to_unvrap"):
            strong_signals.append(summary["relevance_to_unvrap"])

    brief = {
        "number_of_papers_considered": len(summary_packages),
        "recurring_phobias": [item for item, _ in phobias.most_common(5)],
        "recurring_methods": [item for item, _ in methods.most_common(5)],
        "common_outcomes": concise_list_from_sentences(outcomes[:8], fallback="Not reported"),
        "common_limitations": concise_list_from_sentences(limitations[:8], fallback="Not reported"),
        "strongest_signals_for_unvrap": concise_list_from_sentences(strong_signals[:6], fallback="Not reported"),
        "research_gaps": [
            "Many local summaries require manual validation of authors, sample size, and effect sizes.",
            "Clinical strength and statistical significance should be checked in the original papers before implementation decisions.",
            "A later cloud or expert-reviewed phase should improve evidence grounding and metadata extraction.",
        ],
        "implementation_note": "This literature brief was generated in local no-cost mode. It is suitable for a student prototype and internal testing, but UnVRap should manually verify high-stakes research claims.",
        "executive_summary": " ".join(concise_list_from_sentences(outcomes[:4], fallback="Local summaries did not contain enough consistent findings.")),
    }
    package = {
        "generated_at": now_iso(),
        "model_name": "local-extractive-no-cost",
        "brief": brief,
    }
    save_json(BRIEF_FILE, package)
    return package


def batch_generate_all_summaries(
    client: OpenAI,
    vector_store_id: str,
    model_name: str = DEFAULT_MODEL,
) -> List[Dict[str, Any]]:
    packages: List[Dict[str, Any]] = []
    for doc_record in get_all_doc_records():
        package = generate_single_paper_summary(
            client=client,
            vector_store_id=vector_store_id,
            doc_record=doc_record,
            model_name=model_name,
        )
        packages.append(package)
    return packages
