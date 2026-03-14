import json
import torch
from typing import List, Dict, Tuple
from transformers import AutoModelForTokenClassification, AutoTokenizer, AutoModelForSeq2SeqLM

# --- Configuration ---
NER_DIR = "./models/ner_model"
SUMM_DIR = "./models/t5_entity_summ"

# --- Global Model Components (placeholders until loaded) ---
NER_MODEL, NER_TOKENIZER = None, None
SUMM_MODEL, SUMM_TOKENIZER = None, None
ID2LABEL = None
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def load_models():
    """Loads the fine-tuned NER and T5 summarization models."""
    global NER_MODEL, NER_TOKENIZER, SUMM_MODEL, SUMM_TOKENIZER, ID2LABEL
    
    if SUMM_MODEL is not None:
        return
    
    print("Loading NER and Summarization models...")
    
    # NER Model (BioBERT)
    try:
        NER_MODEL = AutoModelForTokenClassification.from_pretrained(NER_DIR, local_files_only=True)
        NER_TOKENIZER = AutoTokenizer.from_pretrained(NER_DIR, local_files_only=True)
        NER_MODEL.to(DEVICE).eval()
        raw_id2label = NER_MODEL.config.id2label
        ID2LABEL = {int(k): v for k, v in raw_id2label.items()}
    except Exception as e:
        print(f"ERROR: Could not load NER model from {NER_DIR}. Run training script first.")
        return

    # Summarization Model (T5)
    try:
        SUMM_MODEL = AutoModelForSeq2SeqLM.from_pretrained(SUMM_DIR, local_files_only=True)
        SUMM_TOKENIZER = AutoTokenizer.from_pretrained(SUMM_DIR, local_files_only=True)
        SUMM_MODEL.to(DEVICE).eval()
    except Exception as e:
        print(f"ERROR: Could not load Summarization model from {SUMM_DIR}. Run training script first.")
        return
        
    print(f"✅ Models loaded successfully on {DEVICE}.")


def extract_entities(text: str) -> List[str]:
    """Runs the trained NER model (Step 2) and extracts clean entity strings."""
    load_models()
    if not NER_MODEL: return []
    
    inputs = NER_TOKENIZER(text, return_tensors="pt", truncation=True)
    inputs = {k: v.to(DEVICE) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = NER_MODEL(**inputs)
        pred_ids = outputs.logits.argmax(dim=-1)[0].tolist()

    tokens = NER_TOKENIZER.convert_ids_to_tokens(inputs["input_ids"][0])
    
    entities = []
    current_tokens = []
    current_type = None

    def clean_token(tok: str) -> str:
        """Cleans BERT wordpieces (e.g., '##pain' -> 'pain')."""
        tok = tok.replace(" ", "")
        if tok.startswith("##"):
            tok = tok[2:]
        return tok

    for tok, tag_id in zip(tokens, pred_ids):
        tag = ID2LABEL.get(tag_id)
        
        if isinstance(tag, bytes): tag = tag.decode("utf-8")
        
        if tag in ["O", None, "[CLS]", "[SEP]"]:
            if current_tokens:
                entities.append(" ".join(current_tokens))
            current_tokens, current_type = [], None
            continue

        prefix, ent_type = tag.split("-", 1)

        if prefix == "B":
            if current_tokens: entities.append(" ".join(current_tokens))
            current_tokens, current_type = [clean_token(tok)], ent_type
        elif prefix == "I" and current_type == ent_type:
            current_tokens.append(clean_token(tok))
        else: # Inconsistent tag
            if current_tokens: entities.append(" ".join(current_tokens))
            current_tokens, current_type = [clean_token(tok)], ent_type
    
    if current_tokens: entities.append(" ".join(current_tokens))
    
    return sorted(set([e for e in entities if e.strip()]))


def evaluate_summary_entities(source: str, summary: str) -> Dict[str, List[str]]:
    """Compares entities in source vs summary using the trained NER model (Step 4 Check)."""
    src_ents = sorted(set(extract_entities(source)))
    sum_ents = sorted(set(extract_entities(summary)))

    covered = [e for e in src_ents if e in sum_ents]
    missing = [e for e in src_ents if e not in sum_ents]
    hallucinated = [e for e in sum_ents if e not in src_ents]

    return {
        "src_entities": src_ents,
        "sum_entities": sum_ents,
        "covered": covered,
        "missing": missing,
        "hallucinated": hallucinated,
    }

def build_entity_prompt(text: str, ents: List[str]) -> str:
    """Creates the input prompt for the T5 model."""
    ents_str = ", ".join(ents) if ents else "none"
    return f"Document: {text}\nEntities: {ents_str}\nSummary:"


def generate_summary_from_prompt(prompt: str, max_length: int = 64, **kwargs) -> str:
    """Generates the summary using the loaded T5 model (Step 3)."""
    load_models()
    if not SUMM_MODEL: return "Model not available."
    
    inputs = SUMM_TOKENIZER(prompt, return_tensors="pt", truncation=True)
    inputs = {k: v.to(DEVICE) for k, v in inputs.items()}

    gen_kwargs = {
        "max_length": max_length,
        "num_beams": kwargs.get("num_beams", 1),
        "do_sample": kwargs.get("do_sample", False),
    }
    if kwargs.get("no_repeat_ngram_size", 0) > 0:
        gen_kwargs["no_repeat_ngram_size"] = kwargs["no_repeat_ngram_size"]

    output_ids = SUMM_MODEL.generate(**inputs, **gen_kwargs)
    return SUMM_TOKENIZER.decode(output_ids[0], skip_special_tokens=True)


def make_safe_fallback_summary(covered_entities: List[str]) -> str:
    """Provides a conservative backup summary based on verified entities."""
    from research.data_generator import get_entity_type, drugs, diseases, symptoms # Import entity definitions
    
    if not covered_entities:
        return "The patient has clinical findings described in the document."
    
    parts = []
    # Note: This classification assumes the entities are from the simple synthetic set
    classified_ents = {e: get_entity_type(e) for e in covered_entities}
    
    def filter_ents(cat): return [e for e, t in classified_ents.items() if t == cat]

    if (d := filter_ents("DISEASE")): parts.append("diagnosed with " + ", ".join(d))
    if (s := filter_ents("SYMPTOM")): parts.append("experiencing " + ", ".join(s))
    if (dr := filter_ents("DRUG")): parts.append("treated with " + ", ".join(dr))
    
    if not parts:
        return "The note contains unspecified clinical information."
    
    return "The patient is " + ", and ".join(parts) + "."

def summarize_with_mitigation(text: str, max_regenerations: int = 1) -> Dict:
    """Full pipeline: NER, T5 generation, and Hallucination mitigation (Week 6)."""
    
    src_entities = sorted(set(extract_entities(text)))

    def one_pass(num_beams=1, no_repeat_ngram_size=0, max_length=64):
        prompt = build_entity_prompt(text, src_entities)
        summary = generate_summary_from_prompt(prompt, max_length=max_length, 
                                               num_beams=num_beams, no_repeat_ngram_size=no_repeat_ngram_size)
        metrics = evaluate_summary_entities(text, summary)
        return summary, metrics

    summary, metrics = one_pass(num_beams=1, no_repeat_ngram_size=0)
    attempts = 1

    if len(metrics["hallucinated"]) == 0:
        return {"summary": summary, "strategy": "baseline", "attempts": attempts, "metrics": metrics}

    # Mitigation Attempt (Regeneration with stricter decoding)
    best_summary, best_metrics = summary, metrics
    for _ in range(max_regenerations):
        attempts += 1
        regenerated_summary, regenerated_metrics = one_pass(num_beams=4, no_repeat_ngram_size=3, max_length=60)
        
        if len(regenerated_metrics["hallucinated"]) <= len(best_metrics["hallucinated"]):
            best_summary, best_metrics = regenerated_summary, regenerated_metrics

        if len(regenerated_metrics["hallucinated"]) == 0:
            return {"summary": regenerated_summary, "strategy": "regenerated", "attempts": attempts, "metrics": regenerated_metrics}

    # Fallback Strategy
    fallback_summary = make_safe_fallback_summary(best_metrics["covered"])
    fallback_metrics = evaluate_summary_entities(text, fallback_summary)
    
    return {"summary": fallback_summary, "strategy": "fallback", "attempts": attempts, "metrics": fallback_metrics}
