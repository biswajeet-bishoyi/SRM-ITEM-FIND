"""
Sentence-Transformers Fine-Tuning Pipeline for SRM CampusFind.
Fine-tunes transformer semantic text representations on SRM campus lost-and-found queries
using contrastive loss and hard negative mining.
"""

import os
import sys
import json
import random

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

OUTPUT_MODEL_DIR = os.path.join(PROJECT_ROOT, "models", "text_encoder", "srm_minilm")


def train_sentence_transformer(
    model_name: str = "all-MiniLM-L6-v2",
    epochs: int = 4,
    batch_size: int = 16
):
    """Fine-tune SentenceTransformer on SRM lost-found query pairs."""
    try:
        from sentence_transformers import SentenceTransformer, InputExample, losses
        from torch.utils.data import DataLoader
        import torch
    except ImportError:
        print("[Fine-Tuning Notice] PyTorch and sentence-transformers are required for deep transformer fine-tuning.")
        print("To run transformer fine-tuning:")
        print("  pip install -r requirements-full.txt")
        print("  python train_text_model.py")
        return False

    print("=" * 60)
    print(f"  Fine-Tuning Transformer Text Encoder: {model_name}")
    print("=" * 60)

    dataset_path = os.path.join(PROJECT_ROOT, "data", "processed", "srm_campusfind_dataset.json")
    if not os.path.exists(dataset_path):
        from src.preprocessing.dataset_generator import generate_dataset
        records = generate_dataset(PROJECT_ROOT)
    else:
        with open(dataset_path, "r", encoding="utf-8") as f:
            records = json.load(f)

    lost_items = [r for r in records if r["status"] == "LOST"]
    found_items = [r for r in records if r["status"] == "FOUND"]
    found_by_pair = {f["pair_id"]: f for f in found_items}

    train_examples = []

    # 1. Positive Training Examples (Semantic score = 1.0)
    for lost in lost_items:
        found = found_by_pair.get(lost["pair_id"])
        if found:
            train_examples.append(InputExample(
                texts=[lost["description"], found["description"]],
                label=1.0
            ))

    # 2. Hard Negatives (Semantic score = 0.0)
    for lost in lost_items:
        same_cat = [f for f in found_items if f["category"] == lost["category"] and f["pair_id"] != lost["pair_id"]]
        if same_cat:
            chosen = random.choice(same_cat)
            train_examples.append(InputExample(
                texts=[lost["description"], chosen["description"]],
                label=0.0
            ))

    print(f"[Fine-Tuning] Built {len(train_examples)} training pairs (Positives + Hard Negatives).")

    # Load base model
    print(f"[Fine-Tuning] Loading base model: {model_name}...")
    model = SentenceTransformer(model_name)

    train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=batch_size)
    train_loss = losses.CosineSimilarityLoss(model=model)

    print(f"[Fine-Tuning] Training for {epochs} epochs...")
    model.fit(
        train_objectives=[(train_dataloader, train_loss)],
        epochs=epochs,
        warmup_steps=10,
        show_progress_bar=True
    )

    os.makedirs(OUTPUT_MODEL_DIR, exist_ok=True)
    model.save(OUTPUT_MODEL_DIR)
    print(f"\n✅ Fine-tuned SentenceTransformer model saved to: {OUTPUT_MODEL_DIR}")
    return True


if __name__ == "__main__":
    train_sentence_transformer()
