"""
Nehasri_task1_2_geneformer.py
Task 1.2: Geneformer foundation-model embedding + Leiden clustering.
Extracted from the submitted notebook. Produces
Nehasri Eragandula_task1_2_geneformer_submission.csv.
Run Nehasri_task1_1_pca.py first in the same session (or this script's own
setup block below re-loads `adata` independently).
"""

import subprocess
import sys

subprocess.run([sys.executable, "-m", "pip", "install"] + "-q pandas>=2.3,<3.1".split(), check=True)
subprocess.run([sys.executable, "-m", "pip", "install"] + "-q scanpy leidenalg python-igraph scikit-learn".split(), check=True)

# from google.colab import drive   # Colab-only
# drive.mount('/content/drive')    # mount manually in Colab, or point ASSIGNMENT_DIR to a local path below

import os, json, time
import numpy as np
import pandas as pd
import scanpy as sc
import anndata as ad
import matplotlib.pyplot as plt

from sklearn.metrics import (
    silhouette_score,
    adjusted_rand_score,
    normalized_mutual_info_score,
)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler

sc.settings.verbosity = 1
sc.settings.set_figure_params(dpi=100, facecolor="white")
np.random.seed(0)

ASSIGNMENT_DIR  = "/content/drive/MyDrive/CS690_Assignment 1"
OUT_DIR         = os.path.join(ASSIGNMENT_DIR, "outputs")
os.makedirs(OUT_DIR, exist_ok=True)

DATA_PATH       = os.path.join(ASSIGNMENT_DIR, "dataset_updated.h5ad")
SAMPLE_SUB_PATH = os.path.join(ASSIGNMENT_DIR, "sample_submission.csv")

if not os.path.exists(DATA_PATH):
    DATA_PATH = os.path.join(ASSIGNMENT_DIR, "dataset.h5ad")

print("Assignment dir:", ASSIGNMENT_DIR)
print("Output dir:", OUT_DIR)
print("Using dataset:", DATA_PATH, "| exists:", os.path.exists(DATA_PATH))

adata = sc.read_h5ad(DATA_PATH)
print("\nCells:", adata.n_obs, "| Genes:", adata.n_vars)
print("X range:", float(adata.X.min()), "to", float(adata.X.max()))

NAME = "Nehasri Eragandula"


GENEFORMER_EMB_PATH = os.path.join(ASSIGNMENT_DIR, "task1_2_geneformer_embeddings.npy")
GENEFORMER_PARTIAL_PATH = GENEFORMER_EMB_PATH + ".partial.npz"
GENEFORMER_DIR = os.path.join(ASSIGNMENT_DIR, "Geneformer")

if os.path.exists(GENEFORMER_EMB_PATH):
    print("Found existing Geneformer embeddings, loading...")
    geneformer_embeddings = np.load(GENEFORMER_EMB_PATH)
    print("Loaded shape:", geneformer_embeddings.shape)
else:
    print("Regenerating Geneformer embeddings...")
    import torch, pickle, sys
    from torch.nn.utils.rnn import pad_sequence
    from torch.utils.data import Dataset, DataLoader
    from transformers import BertForMaskedLM
    from tqdm.auto import tqdm

    # Skip re-clone if it already exists
    if not os.path.exists(GENEFORMER_DIR):
        subprocess.run(["git", "clone", "https://huggingface.co/ctheodoris/Geneformer", GENEFORMER_DIR], check=True)

    MODEL_DIR = os.path.join(GENEFORMER_DIR, "Geneformer-V2-104M")
    sys.path.insert(0, GENEFORMER_DIR)

    with open(os.path.join(GENEFORMER_DIR, "geneformer", "token_dictionary_gc104M.pkl"), "rb") as f:
        token_dictionary = pickle.load(f)
    with open(os.path.join(GENEFORMER_DIR, "geneformer", "gene_name_id_dict_gc104M.pkl"), "rb") as f:
        gene_name_id_dict = pickle.load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    geneformer_model = BertForMaskedLM.from_pretrained(MODEL_DIR, output_hidden_states=True, output_attentions=False)
    geneformer_model.to(device).eval()

    CLS_TOKEN = token_dictionary["<cls>"]
    PAD_TOKEN = token_dictionary["<pad>"]

    X_dense = np.asarray(adata.X.todense() if hasattr(adata.X, "todense") else adata.X)
    gene_names = adata.var_names.tolist()
    ensembl_ids = [gene_name_id_dict.get(g) for g in gene_names]
    valid_gene_idx = [i for i, eid in enumerate(ensembl_ids) if eid in token_dictionary]
    valid_tokens = np.array([token_dictionary[ensembl_ids[i]] for i in valid_gene_idx])

    geneformer_sequences = []
    for row in X_dense:
        expr = row[valid_gene_idx]
        order = np.argsort(-expr)
        order = order[expr[order] > 0]
        seq = [CLS_TOKEN] + valid_tokens[order][:2047].tolist()
        geneformer_sequences.append(seq)

    class GFDataset(Dataset):
        def __init__(self, seqs): self.seqs = seqs
        def __len__(self): return len(self.seqs)
        def __getitem__(self, i): return torch.tensor(self.seqs[i], dtype=torch.long)

    def collate(batch):
        input_ids = pad_sequence(batch, batch_first=True, padding_value=PAD_TOKEN)
        attention_mask = (input_ids != PAD_TOKEN).long()
        return {"input_ids": input_ids, "attention_mask": attention_mask}

    BATCH_SIZE = 8
    loader = DataLoader(GFDataset(geneformer_sequences), batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate)

    embedding_dim = geneformer_model.config.hidden_size
    geneformer_embeddings = np.zeros((adata.n_obs, embedding_dim), dtype=np.float32)

    # Resume from a partial checkpoint if one exists
    start_idx = 0
    if os.path.exists(GENEFORMER_PARTIAL_PATH):
        partial = np.load(GENEFORMER_PARTIAL_PATH)
        saved = partial["embeddings"]
        start_idx = int(partial["idx"])
        geneformer_embeddings[:start_idx] = saved[:start_idx]
        print(f"Resuming from checkpoint at cell {start_idx}/{adata.n_obs}")

    batches_to_skip = start_idx // BATCH_SIZE

    with torch.no_grad():
        idx = start_idx
        for b, batch in enumerate(tqdm(loader, desc="Geneformer embeddings")):
            if b < batches_to_skip:
                continue
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            outputs = geneformer_model(input_ids=input_ids, attention_mask=attention_mask)
            hidden = outputs.hidden_states[-2]
            cls_emb = hidden[:, 0, :].cpu().numpy().astype(np.float32)
            geneformer_embeddings[idx: idx + cls_emb.shape[0]] = cls_emb
            idx += cls_emb.shape[0]

            if b % 100 == 0:
                np.savez(GENEFORMER_PARTIAL_PATH, embeddings=geneformer_embeddings, idx=idx)

    np.save(GENEFORMER_EMB_PATH, geneformer_embeddings)
    if os.path.exists(GENEFORMER_PARTIAL_PATH):
        os.remove(GENEFORMER_PARTIAL_PATH)
    print("Generated + saved Geneformer embeddings:", geneformer_embeddings.shape)

# ---- Geneformer clustering ----
from sklearn.preprocessing import normalize as sk_normalize
geneformer_embeddings_norm = sk_normalize(geneformer_embeddings)

adata_geneformer = sc.AnnData(X=geneformer_embeddings_norm)
adata_geneformer.obs_names = adata.obs_names.astype(str)
adata_geneformer.obsm["X_geneformer"] = geneformer_embeddings_norm

sc.pp.neighbors(adata_geneformer, n_neighbors=15, use_rep="X_geneformer", metric="cosine")

resolutions = [0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50, 0.60, 0.80, 1.00]
gf_res_results = []
for res in resolutions:
    key = f"leiden_r{str(res).replace('.', '_')}"
    sc.tl.leiden(adata_geneformer, resolution=res, key_added=key, random_state=42, flavor="igraph", directed=False)
    labels = adata_geneformer.obs[key].astype(str).to_numpy()
    n_clusters = len(np.unique(labels))
    sil = silhouette_score(adata_geneformer.obsm["X_geneformer"], labels, metric="cosine") if n_clusters > 1 else np.nan
    gf_res_results.append({"resolution": res, "n_clusters": n_clusters, "silhouette": sil})

gf_resolution_df = pd.DataFrame(gf_res_results)
print(gf_resolution_df.to_string(index=False))

best_gf_res = gf_resolution_df.loc[gf_resolution_df["silhouette"].idxmax(), "resolution"]
sc.tl.leiden(adata_geneformer, resolution=float(best_gf_res), key_added="leiden_geneformer_final",
             random_state=42, flavor="igraph", directed=False)
adata_geneformer.obs["leiden_geneformer_final"] = adata_geneformer.obs["leiden_geneformer_final"].astype(str)

geneformer_silhouette = silhouette_score(
    adata_geneformer.obsm["X_geneformer"],
    adata_geneformer.obs["leiden_geneformer_final"].to_numpy(), metric="cosine"
)
print("\nFinal Geneformer resolution:", best_gf_res,
      "| clusters:", adata_geneformer.obs["leiden_geneformer_final"].nunique(),
      "| silhouette:", round(geneformer_silhouette, 4))


# ---- Geneformer UMAP + submission ----
sc.tl.umap(adata_geneformer, min_dist=0.5, spread=1.0, random_state=42)
sc.pl.umap(adata_geneformer, color="leiden_geneformer_final", title="Geneformer Embeddings - Leiden Clusters",
           legend_loc="right margin", frameon=False, show=True)

geneformer_labels_1based = adata_geneformer.obs["leiden_geneformer_final"].astype(int).to_numpy() + 1
geneformer_submission = pd.DataFrame({"ID": np.arange(adata.n_obs), "Label": geneformer_labels_1based})
geneformer_submission_path = os.path.join(ASSIGNMENT_DIR, f"{NAME}_task1_2_geneformer_submission.csv")
geneformer_submission.to_csv(geneformer_submission_path, index=False)
print("Saved:", geneformer_submission_path)

