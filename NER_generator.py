import random, time, json
from pathlib import Path
from collections import Counter
import re
from typing import List, Dict, Tuple
import nltk
# Ensure nltk punkt data is available if running outside of Colab
try:
    nltk.data.find('tokenizers/punkt')
except nltk.downloader.DownloadError:
    nltk.download("punkt", quiet=True)

# --- Entity Definitions (Used for generation and analysis) ---
patients = ["John Smith", "Mary Johnson", "The patient", "He", "She", "Mr. Brown", "Ms. Davis", "Patient A"]
symptoms = ["fever", "high fever", "chest pain", "shortness of breath", "cough", "nausea", "vomiting", "headache"]
diseases = ["diabetes", "type 2 diabetes", "hypertension", "infection", "influenza", "the flu"]
drugs = ["Aspirin", "Paracetamol", "Ibuprofen", "Metformin", "Insulin", "Amoxicillin", "Azithromycin"]
dosages = ["5 mg", "10 mg", "500 mg", "twice daily", "once a day"]
time_phrases = ["yesterday", "last week", "this morning", "2 days ago"]
negations = ["no", "not", "denies", "without"]

# --- Configuration (matching your notebook) ---
small_demo = True
if small_demo:
    TRAIN_N, DEV_N, TEST_N = 200, 50, 50
else:
    TRAIN_N, DEV_N, TEST_N = 50000, 5000, 5000

ADD_DOSAGES = True
ADD_TIME_PHRASES = True
ADD_NEGATION = True

# --- Core Generation Functions ---

def entity_to_bio(entity: str, label: str) -> Tuple[List[str], List[str]]:
    """Converts a phrase into tokens and B-I-O tags."""
    toks = entity.split()
    tags = [f"B-{label}"] + [f"I-{label}"] * (len(toks) - 1)
    return toks, tags

def add_words(tokens: List[str], tags: List[str], words: List[str], tag: str = "O"):
    """Appends words and their corresponding tags to the lists."""
    for w in words:
        tokens.append(w)
        tags.append(tag)

def gen_one() -> Tuple[List[str], List[str]]:
    """Generates a single synthetic clinical example sentence and its BIO tags."""
    tpl = random.choices(
        ["sym_disease", "drug_symptom", "history", "complex", "negation", "dosage"],
        [0.30, 0.25, 0.20, 0.15, 0.05, 0.05]
    )[0]
    tokens, tags = [], []
    p = random.choice(patients)
    s = random.choice(symptoms)
    d = random.choice(diseases)
    dr = random.choice(drugs)
    dose = random.choice(dosages) if ADD_DOSAGES else None
    tp = random.choice(time_phrases) if ADD_TIME_PHRASES else None
    neg = random.choice(negations) if ADD_NEGATION else None

    # --- Templated Sentence Logic (Exactly as in your notebook) ---
    if tpl == "sym_disease":
        if " " in p: et, tg = entity_to_bio(p, "PATIENT"); add_words(tokens, tags, et); tags[-len(et):] = tg
        else: add_words(tokens, tags, [p], "O")
        add_words(tokens, tags, ["has"], "O")
        et, tg = entity_to_bio(s, "SYMPTOM"); add_words(tokens, tags, et); tags[-len(et):] = tg
        add_words(tokens, tags, ["and"], "O")
        et, tg = entity_to_bio(d, "DISEASE"); add_words(tokens, tags, et); tags[-len(et):] = tg
        add_words(tokens, tags, ["."], "O")
    elif tpl == "drug_symptom":
        add_words(tokens, tags, ["The", "doctor", "prescribed"], "O")
        et, tg = entity_to_bio(dr, "DRUG"); add_words(tokens, tags, et); tags[-len(et):] = tg
        add_words(tokens, tags, ["for"], "O")
        et, tg = entity_to_bio(s, "SYMPTOM"); add_words(tokens, tags, et); tags[-len(et):] = tg
        add_words(tokens, tags, ["."], "O")
        if tp and random.random() < 0.25: add_words(tokens, tags, [tp], "O")
    elif tpl == "history":
        if " " in p: et, tg = entity_to_bio(p, "PATIENT"); add_words(tokens, tags, et); tags[-len(et):] = tg
        else: add_words(tokens, tags, [p], "O")
        add_words(tokens, tags, ["has", "a", "history", "of"], "O")
        et, tg = entity_to_bio(d, "DISEASE"); add_words(tokens, tags, et); tags[-len(et):] = tg
        add_words(tokens, tags, ["and", "takes"], "O")
        et, tg = entity_to_bio(dr, "DRUG"); add_words(tokens, tags, et); tags[-len(et):] = tg
        add_words(tokens, tags, ["."], "O")
    elif tpl == "complex":
        if " " in p: et, tg = entity_to_bio(p, "PATIENT"); add_words(tokens, tags, et); tags[-len(et):] = tg
        else: add_words(tokens, tags, [p], "O")
        add_words(tokens, tags, ["complained", "of"], "O")
        et, tg = entity_to_bio(s, "SYMPTOM"); add_words(tokens, tags, et); tags[-len(et):] = tg
        add_words(tokens, tags, [",", "was", "diagnosed", "with"], "O")
        et, tg = entity_to_bio(d, "DISEASE"); add_words(tokens, tags, et); tags[-len(et):] = tg
        add_words(tokens, tags, ["and", "treated", "with"], "O")
        et, tg = entity_to_bio(dr, "DRUG"); add_words(tokens, tags, et); tags[-len(et):] = tg
        if dose and random.random() < 0.3:
            dtoks = dose.split(); add_words(tokens, tags, dtoks, "O")
        add_words(tokens, tags, ["."], "O")
    elif tpl == "negation":
        if " " in p: et, tg = entity_to_bio(p, "PATIENT"); add_words(tokens, tags, et); tags[-len(et):] = tg
        else: add_words(tokens, tags, [p], "O")
        add_words(tokens, tags, [neg], "O")
        et, tg = entity_to_bio(s, "SYMPTOM"); add_words(tokens, tags, et); tags[-len(et):] = tg
        add_words(tokens, tags, ["."], "O")
    else:  # dosage
        add_words(tokens, tags, ["Gave"], "O")
        if dose:
            dtoks = dose.split(); add_words(tokens, tags, dtoks, "O")
        add_words(tokens, tags, ["of"], "O")
        et, tg = entity_to_bio(dr, "DRUG"); add_words(tokens, tags, et); tags[-len(et):] = tg
        if random.random() < 0.6:
            add_words(tokens, tags, ["for"], "O"); et, tg = entity_to_bio(s, "SYMPTOM"); add_words(tokens, tags, et); tags[-len(et):] = tg
        add_words(tokens, tags, ["."], "O")

    if ADD_TIME_PHRASES and random.random() < 0.05:
        add_words(tokens, tags, [tp], "O")

    return tokens, tags

# --- CoNLL File Utilities ---

def write_conll(exs: List[Tuple[List[str], List[str]]], path: Path):
    """Writes the generated examples to a CoNLL format file."""
    with open(path, "w", encoding="utf-8") as f:
        for tokens, tags in exs:
            for t, tag in zip(tokens, tags):
                f.write(f"{t} {tag}\n")
            f.write("\n")

def read_conll_file(path: Path) -> List[Dict]:
    """Reads a .conll file and returns a list of examples."""
    sentences = []
    tokens, tags = [], []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                if tokens:
                    sentences.append({"tokens": tokens, "ner_tags": tags})
                    tokens, tags = [], []
                continue
            parts = line.split()
            # Handle multi-token representation for simplicity in the notebook
            token = " ".join(parts[:-1]) if len(parts) > 2 else parts[0]
            tag = parts[-1]
            tokens.append(token)
            tags.append(tag)
    if tokens:
        sentences.append({"tokens": tokens, "ner_tags": tags})
    return sentences

def conll_to_texts(path: Path) -> List[str]:
    """Converts a CoNLL file back into plain text sentences."""
    texts = []
    for ex in read_conll_file(path):
        # Join tokens, assuming spaces were used for separation
        texts.append(" ".join(ex["tokens"]))
    return texts

def make_dataset_files(out_dir: Path):
    """Generates the full dataset set (TRAIN, DEV, TEST) and writes CoNLL files."""
    out_dir.mkdir(exist_ok=True)
    
    print(f"Starting generation (small_demo={small_demo})...")
    
    def make_set(n, name):
        examples = []
        batch = 1000 if n > 2000 else 100
        t0 = time.time()
        for i in range(n):
            examples.append(gen_one())
            if (i + 1) % batch == 0:
                print(f"{name}: generated {i + 1}/{n} examples ... {time.time() - t0:.1f}s")
        print(f"{name}: done ({n} examples, {time.time() - t0:.1f}s)")
        return examples

    train = make_set(TRAIN_N, "TRAIN")
    dev = make_set(DEV_N, "DEV")
    test = make_set(TEST_N, "TEST")

    write_conll(train, out_dir / "train.conll")
    write_conll(dev, out_dir / "dev.conll")
    write_conll(test, out_dir / "test.conll")
    print(f"\nWrote files to {out_dir.resolve()}")
    
    # Save metadata
    meta = dict(train_n=TRAIN_N, dev_n=DEV_N, test_n=TEST_N, small_demo=small_demo, timestamp=time.time())
    with open(out_dir / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    print("Metadata saved.")
    
    return train, dev, test
