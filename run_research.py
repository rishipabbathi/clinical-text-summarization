from research.data_generator import make_dataset_files, conll_to_texts, TRAIN_N, DEV_N, TEST_N
from research.train_models import train_ner_model, load_ner_dataset, train_summ_model
from research.post_process_eval import evaluate_on_texts, summarize_with_postprocessing, highlight_entities_html, extract_entities
from pathlib import Path
import os
import pandas as pd
from IPython.display import HTML, display

# --- Configuration ---
DATA_DIR = Path("ner_data_quick")

def run_all_research_steps():
    """
    Executes the full pipeline for data generation, model training (Week 3-5), 
    and final evaluation (Week 7).
    """
    
    print("--- 1. Data Generation and Preparation (Week 3) ---")
    
    if not (DATA_DIR / "train.conll").exists():
        print("Data files not found. Please uncomment `make_dataset_files(DATA_DIR)` in run_research.py and run again.")
        return
    
    # Load data for training
    ner_dataset, ner_labels = load_ner_dataset(DATA_DIR)
    print(f"Loaded {len(ner_dataset['train'])} NER training examples.")
    
    print("\n--- 2. Model Training (Weeks 3-5) ---")
    
    print("Training steps skipped. Assuming models are saved in ./models/ for Week 7 evaluation.")

    print("\n--- 3. Entity-Aware Evaluation (Week 7) ---")
    
    test_texts = conll_to_texts(DATA_DIR / "test.conll")
    
    # Run pipeline on a sample text
    sample_text = test_texts[0] if test_texts else "The patient has fever and was given paracetamol."
    result = summarize_with_postprocessing(sample_text)
    
    # Display the sample result with highlighting
    src_ents = extract_entities(sample_text)
    sum_ents = extract_entities(result["summary"])
    
    print("\n--- Sample Pipeline Result ---")
    print(f"Source Strategy: {result['strategy']}")
    print("Source:")
    print(highlight_entities_html(sample_text, src_ents))
    print("Summary:")
    print(highlight_entities_html(result["summary"], sum_ents))
    
    # Run full evaluation on test set (requires trained models)
    if os.path.exists("./models/ner_model") and os.path.exists("./models/t5_entity_summ"):
        print("\n--- 4. Automatic Evaluation on Test Set ---")
        df_eval = evaluate_on_texts(test_texts)
        print("\nEvaluation Head:")
        print(df_eval[["source", "summary", "hallucination_rate", "strategy"]].head())
    else:
        print("\nSkipping full test set evaluation: Trained models not found in ./models/")

if __name__ == "__main__":
    run_all_research_steps()
