import re
import json
import pandas as pd
from typing import List, Dict

# Import core pipeline logic
from research.inference_core import summarize_with_mitigation, extract_entities
from research.data_generator import conll_to_texts 

# --- Week 7: Post-Processing ---

def clean_summary_text(text: str) -> str:
    """Simple linguistic cleanup (capitalization, punctuation)."""
    if not text: return text
    
    text = text.strip()
    text = re.sub(r"\s+", " ", text)
    if len(text) > 0:
        text = text[0].upper() + text[1:]
    if text and text[-1] not in ".!?":
        text = text + "."
    return text

def summarize_with_postprocessing(text: str, max_regenerations: int = 1) -> Dict:
    """Runs the full pipeline and applies post-processing."""
    base = summarize_with_mitigation(text, max_regenerations=max_regenerations)

    raw_summary = base["summary"]
    cleaned = clean_summary_text(raw_summary)

    base["summary_raw"] = raw_summary
    base["summary"] = cleaned
    return base

# --- Week 7: Entity Highlighting (for Python display) ---

ENTITY_COLORS = {
    "PATIENT": "#d1c4e9", "SYMPTOM": "#ffcdd2", "DISEASE": "#ffe0b2", 
    "DRUG": "#bbdefb", "OTHER": "#c8e6c9",
}

def infer_entity_type_for_highlight(ent: str) -> str:
    """Infers entity type using simple rules (for display only)."""
    # This requires using the rule-based definitions if the real NER model doesn't output types
    from research.data_generator import patients, symptoms, diseases, drugs
    e = ent.lower()
    if e in [p.lower() for p in patients]: return "PATIENT"
    if e in [p.lower() for p in symptoms]: return "SYMPTOM"
    if e in [p.lower() for p in diseases]: return "DISEASE"
    if e in [p.lower() for p in drugs]: return "DRUG"
    return "OTHER"

def highlight_entities_html(text: str, entities: List[str]) -> str:
    """Returns HTML string with entities highlighted for console/Gradio display."""
    from html import escape
    
    html_text = escape(text)
    entities_sorted = sorted(set(entities), key=len, reverse=True)

    for ent in entities_sorted:
        if not ent.strip(): continue
        ent_type = infer_entity_type_for_highlight(ent)
        color = ENTITY_COLORS.get(ent_type, ENTITY_COLORS["OTHER"])

        pattern = r"\b" + re.escape(ent) + r"\b"
        span = (
            f'<span style="background-color:{color}; padding:1px 3px; border-radius:3px;'
            f' font-weight:bold;">{escape(ent)}</span>'
        )
        html_text = re.sub(pattern, span, html_text, flags=re.IGNORECASE)

    return html_text

# --- Week 7: Automatic Evaluation ---

def evaluate_on_texts(texts: List[str], max_regenerations: int = 1) -> pd.DataFrame:
    """Runs the full pipeline on a set of texts and returns aggregated entity metrics."""
    rows = []

    for i, src in enumerate(texts, start=1):
        res = summarize_with_postprocessing(src, max_regenerations=max_regenerations)
        m = res["metrics"]

        src_ents = m.get("src_entities", [])
        sum_ents = m.get("sum_entities", [])
        
        rows.append({
            "id": i,
            "source": src,
            "summary": res["summary"],
            "strategy": res.get("strategy", "unknown"),
            "src_entity_count": len(src_ents),
            "sum_entity_count": len(sum_ents),
            "covered_count": len(m.get("covered", [])),
            "missing_count": len(m.get("missing", [])),
            "hallucinated_count": len(m.get("hallucinated", [])),
        })

    df = pd.DataFrame(rows)

    df["coverage_rate"] = df["covered_count"] / df["src_entity_count"].replace(0, 1)
    df["hallucination_rate"] = df["hallucinated_count"] / df["sum_entity_count"].replace(0, 1)

    print("=== Aggregate metrics (entity-level) ===")
    print("Mean coverage rate: ", df["coverage_rate"].mean())
    print("Mean hallucination rate: ", df["hallucination_rate"].mean())
    print("\nStrategy counts:")
    print(df["strategy"].value_counts())

    return df
