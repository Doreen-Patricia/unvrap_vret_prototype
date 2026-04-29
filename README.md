# UnVRap VRET Research Summarizer - Student Prototype

This starter project gives you a clear cloud-path prototype for the UnVRap project:

1. Upload all UnVRap papers
2. Store them in an OpenAI vector store
3. Generate one structured summary per paper
4. Inspect evidence chunks
5. Generate a literature brief across all saved summaries

## Recommended stack

- Python 3.11+
- PyCharm
- Streamlit for the prototype UI
- OpenAI API for vector store indexing, retrieval, and summary generation

## Project files

- `app.py` - Streamlit prototype UI
- `prototype_core.py` - shared project logic
- `bulk_upload.py` - command line upload script for a whole folder
- `batch_summarize.py` - command line batch summarizer
- `evaluation_template.csv` - review sheet for testing with UnVRap

## Step 1: Open in PyCharm

Create a new PyCharm project or open this folder directly.

## Step 2: Create and activate a virtual environment

### Windows

```bash
python -m venv .venv
.venv\Scripts\activate
```

### macOS / Linux

```bash
python3 -m venv .venv
source .venv/bin/activate
```

## Step 3: Install dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

## Step 4: Add your API key

Copy `.env.example` to `.env` and replace the placeholder key.

### Windows PowerShell

```powershell
copy .env.example .env
```

### macOS / Linux

```bash
cp .env.example .env
```

Then edit `.env`:

```env
OPENAI_API_KEY=your_real_key_here
MODEL_NAME=gpt-4.1-mini
```

## Step 5: Put UnVRap files in a folder

For command-line bulk upload, create a folder such as:

```text
data/uploads_from_unvrap/
```

Put all shared PDF and DOCX files there.

Try to keep filenames unique, for example:

- `garcia_2023_spider_vret_rct.pdf`
- `smith_2022_flying_phobia_vr.docx`

## Step 6A: Upload all papers with the command line

This is the clearest way to ingest a whole folder.

```bash
python bulk_upload.py --folder data/uploads_from_unvrap --vector-store-name unvrap-vret-kb
```

What happens:

- the script creates or reuses a vector store
- uploads each file to OpenAI
- adds metadata for filtering
- writes project state to `outputs/project_state.json`
- writes a manifest to `outputs/upload_manifest.json`

If you already have a vector store ID and want to keep using it:

```bash
python bulk_upload.py --folder data/uploads_from_unvrap --vector-store-id vs_your_existing_id
```

## Step 6B: Or upload through the Streamlit app

```bash
streamlit run app.py
```

Inside the app:

1. Create or reuse a vector store in the sidebar
2. Go to **Upload & index**
3. Drag in all PDFs/DOCX files
4. Click **Upload + index in OpenAI**
5. Confirm the manifest table shows your files

## Step 7: Test one paper first

In the app:

1. Open **Single paper summary**
2. Select one uploaded paper
3. Click **Generate selected paper summary**
4. Click **Show evidence chunks**
5. Check whether the summary matches the paper

Do this for 2 to 3 papers before running the full batch.

## Step 8: Batch summarize all papers

### In the app

Use the **Batch summarize all uploaded papers** button.

### Or from the terminal

```bash
python batch_summarize.py
```

This creates one JSON summary file per paper in:

```text
outputs/paper_summaries/
```

## Step 9: Generate the literature brief

After individual summaries exist, return to the app and open **Literature brief**.

Click **Generate literature brief**.

This creates:

```text
outputs/literature_brief.json
```

## Step 10: Test with the evaluation sheet

Open `evaluation_template.csv` in Excel or Google Sheets and score each summary on:

- factual accuracy
- completeness
- conciseness
- evidence traceability
- usefulness

Start with 5 to 10 papers.

## Suggested student workflow

### Week 1

- set up PyCharm
- confirm API key works
- upload 3 sample papers
- generate 1 summary successfully

### Week 2

- refine summary schema
- test 5 papers
- fill in evaluation sheet
- improve prompt wording

### Week 3

- upload all available UnVRap sample papers
- batch summarize
- generate literature brief
- collect reviewer feedback

### Week 4

- improve weak fields
- prepare screenshots and outputs for report/presentation
- document limitations and next implementation steps

## What makes this a working prototype

You have a working prototype when you can do all of the following:

- upload multiple UnVRap papers
- store them in a vector store
- generate a structured JSON summary for a selected paper
- show evidence chunks for review
- batch summarize all papers
- generate one literature brief across all paper summaries

## Next step for later implementation

After this prototype works, the next implementation version should add:

- user authentication
- reviewer approval workflow
- duplicate detection by DOI/title
- better metadata extraction
- export to dashboard or internal knowledge base
- logging and usage tracking
