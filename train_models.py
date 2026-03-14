import json
import os
import numpy as np
from pathlib import Path
from datasets import Dataset, DatasetDict
from transformers import (
    AutoTokenizer, AutoModelForTokenClassification, AutoModelForSeq2SeqLM,
    TrainingArguments, Trainer, DataCollatorForTokenClassification, DataCollatorForSeq2Seq
)
from seqeval.metrics import f1_score, precision_score, recall_score
import torch

# Import data generation utilities
from research.data_generator import read_conll_file, conll_to_texts 

# --- Configuration ---
OUT_DIR = Path("ner_data_quick") # Should be created by data_generator.py
NER_MODEL_DIR = "./models/ner_model"
SUMM_MODEL_DIR = "./models/t5_entity_summ"
os.makedirs(NER_MODEL_DIR, exist_ok=True)
os.makedirs(SUMM_MODEL_DIR, exist_ok=True)
os.environ["WANDB_DISABLED"] = "true" # Disable external logging for local run

# --- 1. NER Training Functions (Weeks 3) ---

def load_ner_dataset(data_dir: Path) -> Tuple[DatasetDict, List[str]]:
    """Loads CoNLL files and converts to Hugging Face DatasetDict."""
    train_examples = read_conll_file(data_dir / "train.conll")
    dev_examples = read_conll_file(data_dir / "dev.conll")
    test_examples = read_conll_file(data_dir / "test.conll")
    
    all_tags = sorted(list({t for ex in (train_examples + dev_examples + test_examples) for t in ex["ner_tags"]}))
    label_to_id = {label: i for i, label in enumerate(all_tags)}
    
    def convert_tags(example):
        return {
            "tokens": example["tokens"],
            "ner_tags": [label_to_id[t] for t in example["ner_tags"]]
        }
        
    dataset = DatasetDict({
        "train": Dataset.from_list([convert_tags(e) for e in train_examples]),
        "validation": Dataset.from_list([convert_tags(e) for e in dev_examples]),
        "test": Dataset.from_list([convert_tags(e) for e in test_examples]),
    })
    return dataset, all_tags

def tokenize_and_align_labels(batch, tokenizer, label_list):
    """Aligns NER tags with subword tokens (Crucial step from your notebook)."""
    label_to_id = {l: i for i, l in enumerate(label_list)}
    tokenized = tokenizer(batch["tokens"], is_split_into_words=True, truncation=True)
    labels = []
    
    for i, ner_tags in enumerate(batch["ner_tags"]):
        word_ids = tokenized.word_ids(batch_index=i)
        label_ids = []
        prev = None
        for w in word_ids:
            if w is None:
                label_ids.append(-100) # Special token mask
            else:
                if w != prev:
                    label_ids.append(ner_tags[w])
                else:
                    label_ids.append(-100)
            prev = w
        labels.append(label_ids)
    
    tokenized["labels"] = labels
    return tokenized

def compute_ner_metrics(p):
    """Computes NER evaluation metrics (precision, recall, F1) using seqeval."""
    preds, labs = p
    pred_ids = np.argmax(preds, axis=2)
    
    # Load label list dynamically if needed, or pass it:
    label_list = json.load(open(os.path.join(NER_MODEL_DIR, "ner_label_list.json"))) 

    t_preds, t_labs = [], []
    for pr, lb in zip(pred_ids, labs):
        cp, cl = [], []
        for p_i, l_i in zip(pr, lb):
            if l_i != -100:
                cp.append(label_list[p_i])
                cl.append(label_list[l_i])
        t_preds.append(cp)
        t_labs.append(cl)

    return {
        "precision": precision_score(t_labs, t_preds),
        "recall": recall_score(t_labs, t_preds),
        "f1": f1_score(t_labs, t_preds),
    }

def train_ner_model(dataset: DatasetDict, label_list: List[str]):
    """Sets up and trains the BERT-based NER model."""
    ner_tokenizer = AutoTokenizer.from_pretrained("bert-base-cased", use_fast=True)
    
    tokenized_datasets = dataset.map(
        lambda batch: tokenize_and_align_labels(batch, ner_tokenizer, label_list), 
        batched=True
    )
    
    # Save label list for prediction step later
    with open(os.path.join(NER_MODEL_DIR, "ner_label_list.json"), "w") as f:
        json.dump(label_list, f, indent=2)

    model = AutoModelForTokenClassification.from_pretrained("bert-base-cased", num_labels=len(label_list))
    
    args = TrainingArguments(
        output_dir=NER_MODEL_DIR,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        num_train_epochs=5,
        learning_rate=2e-5,
        logging_steps=20,
        report_to="none"
    )
    data_collator = DataCollatorForTokenClassification(ner_tokenizer)
    
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["validation"],
        tokenizer=ner_tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_ner_metrics
    )
    
    print("Starting NER Training...")
    trainer.train()
    trainer.save_model(NER_MODEL_DIR)
    ner_tokenizer.save_pretrained(NER_MODEL_DIR)
    print(f"🎉 NER training complete and model saved to {NER_MODEL_DIR}!")

    
# --- 2. T5 Summarization Training Functions (Weeks 4-5) ---

def make_training_example(text: str, ner_inference_fn):
    """Creates the input/summary pair for T5 training (Document + Entities -> Summary)."""
    # Note: In a real scenario, you would need the *trained* NER model here.
    # For this script, we'll use a placeholder/simplified entity extraction function.
    
    # Simplified entity extraction using rule-based/mock for the prompt construction
    from research.data_generator import entity_to_bio # Reuse definitions
    
    # This is a critical point: The T5 model is trained to summarize based on the NER output.
    # We must use the *mock* entity generation here as done in your notebook:
    
    # Mock entity generation for training data (to match the notebook's approach)
    mock_ents = [p.lower() for p in text.split() if p.lower() in [e.lower() for e in drugs + diseases + symptoms]]

    # Simplified summary to match the original notebook's synthetic target
    summary = "Patient with " + ", ".join(mock_ents) + "."
    
    prompt = f"Document: {text}\nEntities: {', '.join(mock_ents)}\nSummary:"
    return {"input": prompt, "summary": summary}

def tokenize_summ_fn(batch, summ_tokenizer):
    """Tokenizes inputs and targets for the T5 sequence-to-sequence task."""
    model_inputs = summ_tokenizer(batch["input"], max_length=512, truncation=True)
    with summ_tokenizer.as_target_tokenizer():
        labels = summ_tokenizer(batch["summary"], max_length=64, truncation=True)
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

def train_summ_model(ner_data_dir: Path):
    """Sets up and trains the T5 summarization model."""
    
    # Load texts from the generated data
    train_texts = conll_to_texts(ner_data_dir / "train.conll")
    dev_texts = conll_to_texts(ner_data_dir / "dev.conll")
    
    # Prepare T5 training data
    train_data = [make_training_example(t, None) for t in train_texts]
    dev_data = [make_training_example(t, None) for t in dev_texts]

    dataset = Dataset.from_list(train_data).train_test_split(test_size=0.2)
    valset = Dataset.from_list(dev_data)
    
    dataset = DatasetDict({
        "train": dataset["train"],
        "validation": valset
    })
    
    # Load T5 components
    model_name = "t5-small"
    summ_tokenizer = AutoTokenizer.from_pretrained(model_name)
    summ_model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    
    tokenized_datasets = dataset.map(lambda batch: tokenize_summ_fn(batch, summ_tokenizer), batched=True)

    args = TrainingArguments(
        output_dir=SUMM_MODEL_DIR,
        per_device_train_batch_size=2,
        per_device_eval_batch_size=2,
        learning_rate=5e-5,
        num_train_epochs=5,
        weight_decay=0.01,
        logging_steps=20,
        report_to="none",
    )
    
    data_collator = DataCollatorForSeq2Seq(tokenizer=summ_tokenizer, model=summ_model)
    
    trainer = Trainer(
        model=summ_model,
        args=args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["validation"],
        tokenizer=summ_tokenizer,
        data_collator=data_collator,
    )
    
    print("\nStarting T5 Summarization Training...")
    trainer.train()
    trainer.save_model(SUMM_MODEL_DIR)
    summ_tokenizer.save_pretrained(SUMM_MODEL_DIR)
    print(f"🎉 T5 training complete and model saved to {SUMM_MODEL_DIR}!")

if __name__ == "__main__":
    # 1. Generate Data
    # For a real run, you would ensure models/ and ner_data_quick/ exist
    
    # The actual data generation logic is run here:
    # from research.data_generator import make_dataset_files
    # make_dataset_files(OUT_DIR)
    
    # 2. Train NER Model
    # ner_dataset, ner_labels = load_ner_dataset(OUT_DIR)
    # train_ner_model(ner_dataset, ner_labels)

    # 3. Train Summarization Model
    # train_summ_model(OUT_DIR)
    print("\nTo run training, please uncomment the function calls in __main__.")
    print("Make sure you have downloaded the Hugging Face models first.")
