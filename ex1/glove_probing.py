"""
Exercice 1 - Embeddings statiques : GloVe moyenné + Logistic Regression
On fait la moyenne des vecteurs GloVe de chaque mot du doc.
"""

import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from utils import (seed_everything, load_ag_news_subset,
                   compute_classification_metrics,
                   load_needed_glove_vectors)
import numpy as np
from sklearn.linear_model import LogisticRegression

SEEDS = [42, 123, 456]
TRAIN_SIZE = 4000
TEST_SIZE = 1000
GLOVE_DIM = 100


def text_to_glove_vector(text, glove_vectors, dim=100):
    """Moyenne des vecteurs GloVe pour les mots du texte."""
    tokens = text.lower().split()
    vectors = []
    for token in tokens:
        if token in glove_vectors:
            vectors.append(glove_vectors[token])
    if vectors:
        return np.mean(vectors, axis=0)
    else:
        return np.zeros(dim)


def main():
    # on charge d'abord les données pour connaitre le vocabulaire
    train_texts, train_labels, test_texts, test_labels = load_ag_news_subset(
        train_size=TRAIN_SIZE, test_size=TEST_SIZE, seed=42
    )

    # vocabulaire de tous les textes
    all_words = set()
    for t in train_texts + test_texts:
        all_words.update(t.lower().split())

    print("Chargement de GloVe...")
    glove_vectors, glove_dim = load_needed_glove_vectors(all_words)
    print(f"  {len(glove_vectors)} vecteurs chargés (dim={glove_dim})")

    all_results = []

    for seed in SEEDS:
        seed_everything(seed)
        print(f"\n--- Seed {seed} ---")

      
        print("Vectorisation des textes...")
        X_train = np.array([text_to_glove_vector(t, glove_vectors, GLOVE_DIM) for t in train_texts])
        X_test = np.array([text_to_glove_vector(t, glove_vectors, GLOVE_DIM) for t in test_texts])

        
        clf = LogisticRegression(max_iter=1000, random_state=seed)
        clf.fit(X_train, train_labels)

        y_pred = clf.predict(X_test)
        y_proba = clf.predict_proba(X_test)

        metrics = compute_classification_metrics(test_labels, y_pred, y_proba)
        all_results.append(metrics)

        for k, v in metrics.items():
            print(f"  {k}: {v:.4f}")

    print("\n" + "="*60)
    print("GloVe + Probing — Résultats (mean ± std)")
    print("="*60)
    for key in all_results[0]:
        vals = [r[key] for r in all_results]
        print(f"  {key}: {np.mean(vals):.4f} ± {np.std(vals):.4f}")


if __name__ == "__main__":
    main()
