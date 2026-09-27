# Legacy Notebook Modernizer

Built for the IBM Bob 2.0 Hackathon.

IBM Bob IDE analyzes messy legacy Jupyter notebooks, catches real production-blocking bugs, and refactors one into a tested, deployable FastAPI service.

## Live Demo

https://legacy-notebook-modernizer-1.onrender.com/docs

## What This Does

1. **Analysis** — Bob reads a legacy notebook and produces a plain-English explanation plus a severity-rated list of issues (see `analyzer/`)
2. **Refactor** — Bob rebuilds the notebook into a production-ready FastAPI service (see `app/`), fixing the bugs it found while preserving the original model's exact behavior
3. **Tests** — Bob writes a full pytest suite validating the service (see `tests/`)

## Real Bugs Found

- **Data leakage**: `pd.get_dummies()` applied separately to train and test data, which silently misaligns columns at inference time
- **Silent logic bug**: a missing assignment left every passenger over 64 with a wrong feature value, with no error raised
- **Misleading accuracy**: a "99% accuracy" notebook defined a train/test split but never used it

## Running Locally

\`\`\`bash
pip install -r requirements.txt
python -m app.model      # trains and saves the pipeline
uvicorn app.main:app --reload
\`\`\`

Then visit `http://localhost:8000/docs`.

## Project Structure

\`\`\`
sample_notebooks/   # the three original messy notebooks + dataset
analyzer/            # Bob's analysis reports for each notebook
app/                 # the refactored FastAPI service
tests/               # pytest suite (18/18 passing)
bob_sessions/        # Bob IDE task session summaries
\`\`\`

## Built With

IBM Bob IDE, Python, FastAPI, scikit-learn, pandas, Pydantic, pytest
