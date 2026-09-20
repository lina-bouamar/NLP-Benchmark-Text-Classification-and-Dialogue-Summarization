"""
Fonctions utilitaires partagées entre tous les scripts du projet.
Seeding, métriques, chargement des données, GloVe, embedding similarity.
"""

import os
import random
import zipfile
from urllib.request import urlretrieve

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from transformers import AutoTokenizer, AutoModel


def seed_everything(seed):
    """Fixe toutes les sources d'aléa pour la reproductibilité."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ========== Classification metrics (Exercice 1) ==========

def compute_classification_metrics(y_true, y_pred, y_proba=None, num_classes=4):
    """Calcule toutes les métriques de l'exercice 1."""
    metrics = {}
    metrics["micro_acc"] = accuracy_score(y_true, y_pred)

    # macro accuracy = moyenne des accuracy par classe
    per_class_acc = []
    for cls in range(num_classes):
        mask = np.array(y_true) == cls
        if mask.sum() > 0:
            per_class_acc.append(accuracy_score(
                np.array(y_true)[mask], np.array(y_pred)[mask]
            ))
    metrics["macro_acc"] = np.mean(per_class_acc)

    metrics["micro_f1"] = f1_score(y_true, y_pred, average="micro")
    metrics["macro_f1"] = f1_score(y_true, y_pred, average="macro")

    if y_proba is not None:
        try:
            metrics["auroc"] = roc_auc_score(
                y_true, y_proba, multi_class="ovr", average="macro"
            )
        except ValueError:
            metrics["auroc"] = float("nan")
    else:
        metrics["auroc"] = float("nan")

    return metrics


# ========== Data loading ==========

def load_ag_news_subset(train_size=4000, test_size=1000, seed=42):
    """Charge AG News et sous-échantillonne avec un seed fixe."""
    from datasets import load_dataset
    ds = load_dataset("fancyzhx/ag_news")

    rng = np.random.RandomState(seed)
    train_idx = rng.choice(len(ds["train"]), size=train_size, replace=False)
    test_idx = rng.choice(len(ds["test"]), size=test_size, replace=False)

    train_texts = [ds["train"][int(i)]["text"] for i in train_idx]
    train_labels = [ds["train"][int(i)]["label"] for i in train_idx]
    test_texts = [ds["test"][int(i)]["text"] for i in test_idx]
    test_labels = [ds["test"][int(i)]["label"] for i in test_idx]

    return train_texts, train_labels, test_texts, test_labels


# ========== GloVe ==========

GLOVE_URL = "https://nlp.stanford.edu/data/glove.6B.zip"
GLOVE_DIR = "data/glove"
GLOVE_ZIP_PATH = os.path.join(GLOVE_DIR, "glove.6B.zip")
GLOVE_TXT_PATH = os.path.join(GLOVE_DIR, "glove.6B.100d.txt")


def download_glove():
    """Télécharge et extrait GloVe si pas déjà fait."""
    os.makedirs(GLOVE_DIR, exist_ok=True)
    if not os.path.exists(GLOVE_ZIP_PATH):
        print("Downloading GloVe archive from Stanford...")
        urlretrieve(GLOVE_URL, GLOVE_ZIP_PATH)
        print("Download complete.")
    else:
        print("Archive already present:", GLOVE_ZIP_PATH)

    if not os.path.exists(GLOVE_TXT_PATH):
        print("Extracting glove.6B.100d.txt ...")
        with zipfile.ZipFile(GLOVE_ZIP_PATH, "r") as zf:
            zf.extract("glove.6B.100d.txt", path=GLOVE_DIR)
        print("Extraction complete.")
    else:
        print("Text file already present:", GLOVE_TXT_PATH)


def load_needed_glove_vectors(vocab):
    """Charge les vecteurs GloVe uniquement pour les mots du vocabulaire."""
    download_glove()
    vectors = {}
    dim = None
    with open(GLOVE_TXT_PATH, "r", encoding="utf-8", newline="\n", errors="ignore") as f:
        for line in f:
            parts = line.rstrip().split(" ")
            word = parts[0]
            values = parts[1:]
            if dim is None:
                dim = len(values)
            if word in vocab:
                vectors[word] = np.asarray(values, dtype=np.float32)
    return vectors, dim


def build_glove_embedding_matrix(stoi, glove_vectors, dim, random_scale=0.6):
    """Construit la matrice d'embeddings alignée sur notre vocabulaire."""
    matrix = np.random.normal(scale=random_scale, size=(len(stoi), dim)).astype(np.float32)
    matrix[0] = 0  # padding
    found = 0
    for word, idx in stoi.items():
        if word in glove_vectors:
            matrix[idx] = glove_vectors[word]
            found += 1
    coverage = found / len(stoi)
    return matrix, coverage


# ========== ROUGE ==========

def compute_rouge_n(candidate, reference, n=1):
    """Calcule ROUGE-N (F1) entre deux strings."""
    def get_ngrams(tokens, n):
        return [tuple(tokens[i:i+n]) for i in range(len(tokens)-n+1)]

    cand_tokens = candidate.lower().split()
    ref_tokens = reference.lower().split()

    if len(cand_tokens) < n or len(ref_tokens) < n:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}

    cand_ngrams = get_ngrams(cand_tokens, n)
    ref_ngrams = get_ngrams(ref_tokens, n)

    ref_counts = {}
    for ng in ref_ngrams:
        ref_counts[ng] = ref_counts.get(ng, 0) + 1

    cand_counts = {}
    for ng in cand_ngrams:
        cand_counts[ng] = cand_counts.get(ng, 0) + 1

    matches = 0
    for ng, count in cand_counts.items():
        matches += min(count, ref_counts.get(ng, 0))

    precision = matches / len(cand_ngrams) if cand_ngrams else 0.0
    recall = matches / len(ref_ngrams) if ref_ngrams else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

    return {"precision": precision, "recall": recall, "f1": f1}


def lcs_length(a, b):
    """Plus longue sous-séquence commune."""
    m, n = len(a), len(b)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if a[i-1] == b[j-1]:
                dp[i][j] = dp[i-1][j-1] + 1
            else:
                dp[i][j] = max(dp[i-1][j], dp[i][j-1])
    return dp[m][n]


def compute_rouge_l(candidate, reference):
    """Calcule ROUGE-L basé sur la LCS."""
    cand_tokens = candidate.lower().split()
    ref_tokens = reference.lower().split()

    if not cand_tokens or not ref_tokens:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}

    lcs = lcs_length(cand_tokens, ref_tokens)
    precision = lcs / len(cand_tokens) if cand_tokens else 0.0
    recall = lcs / len(ref_tokens) if ref_tokens else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

    return {"precision": precision, "recall": recall, "f1": f1}


# ========== Embedding similarity ==========

ST_NAME = "sentence-transformers/all-MiniLM-L6-v2"
_st_tokenizer = None
_st_encoder = None


def _load_st_model(device):
    """Charge le modèle de sentence embedding (lazy loading)."""
    global _st_tokenizer, _st_encoder
    if _st_tokenizer is None:
        _st_tokenizer = AutoTokenizer.from_pretrained(ST_NAME)
        _st_encoder = AutoModel.from_pretrained(ST_NAME).to(device).eval()
    return _st_tokenizer, _st_encoder


@torch.no_grad()
def encode_sentences(texts, device=None, batch_size=32, max_length=128):
    """Encode des phrases en vecteurs normalisés (mean pooling + L2 norm)."""
    if device is None:
        device = get_device()
    tokenizer, encoder = _load_st_model(device)

    all_emb = []
    for start in range(0, len(texts), batch_size):
        chunk = texts[start:start + batch_size]
        enc = tokenizer(
            chunk, padding=True, truncation=True,
            max_length=max_length, return_tensors="pt"
        ).to(device)
        out = encoder(**enc)
        mask = enc["attention_mask"].unsqueeze(-1).float()
        pooled = (out.last_hidden_state * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)
        pooled = F.normalize(pooled, p=2, dim=-1)
        all_emb.append(pooled.cpu().numpy())
    return np.vstack(all_emb)


def compute_embedding_similarity(candidates, references, device=None):
    """Cosinus entre sentence embeddings (comme dans TP3)."""
    if device is None:
        device = get_device()
    cand_embs = encode_sentences(candidates, device)
    ref_embs = encode_sentences(references, device)
    sims = (cand_embs * ref_embs).sum(axis=1)
    return sims
