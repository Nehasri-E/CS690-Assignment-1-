"""
Nehasri_task1_2_scgpt.py
Task 1.2: scGPT foundation-model embedding + Leiden clustering.
Extracted from the submitted notebook. Produces
Nehasri Eragandula_task1_2_scgpt_submission.csv.
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


# ---- scGPT clustering ----
adata_scgpt_emb = sc.AnnData(X=scgpt_embeddings)
adata_scgpt_emb.obs_names = adata.obs_names.copy()

sc.pp.neighbors(adata_scgpt_emb, n_neighbors=15, use_rep="X", metric="euclidean")

resolutions = [0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50, 0.60, 0.80, 1.00]
scgpt_res_results = []
for res in resolutions:
    key = f"leiden_{res}"
    sc.tl.leiden(adata_scgpt_emb, resolution=res, key_added=key, random_state=42, flavor="igraph", directed=False)
    labels = adata_scgpt_emb.obs[key].astype(str).to_numpy()
    n_clusters = len(np.unique(labels))
    sil = silhouette_score(adata_scgpt_emb.X, labels, metric="euclidean") if n_clusters > 1 else np.nan
    scgpt_res_results.append({"resolution": res, "n_clusters": n_clusters, "silhouette": sil})

scgpt_resolution_df = pd.DataFrame(scgpt_res_results)
print(scgpt_resolution_df.to_string(index=False))

best_scgpt_res = scgpt_resolution_df.loc[scgpt_resolution_df["silhouette"].idxmax(), "resolution"]
sc.tl.leiden(adata_scgpt_emb, resolution=float(best_scgpt_res), key_added="leiden_scgpt_final",
             random_state=42, flavor="igraph", directed=False)
adata_scgpt_emb.obs["leiden_scgpt_final"] = adata_scgpt_emb.obs["leiden_scgpt_final"].astype(str)

scgpt_silhouette = silhouette_score(
    adata_scgpt_emb.X, adata_scgpt_emb.obs["leiden_scgpt_final"].to_numpy(), metric="euclidean"
)
print("\nFinal scGPT resolution:", best_scgpt_res,
      "| clusters:", adata_scgpt_emb.obs["leiden_scgpt_final"].nunique(),
      "| silhouette:", round(scgpt_silhouette, 4))


# ---- scGPT UMAP + submission ----
sc.tl.umap(adata_scgpt_emb, min_dist=0.5, spread=1.0, random_state=42)
sc.pl.umap(adata_scgpt_emb, color="leiden_scgpt_final", title="scGPT Embeddings - Leiden Clusters",
           legend_loc="right margin", frameon=False, show=True)

scgpt_labels_1based = adata_scgpt_emb.obs["leiden_scgpt_final"].astype(int).to_numpy() + 1
scgpt_submission = pd.DataFrame({"ID": np.arange(adata.n_obs), "Label": scgpt_labels_1based})
scgpt_submission_path = os.path.join(ASSIGNMENT_DIR, f"{NAME}_task1_2_scgpt_submission.csv")
scgpt_submission.to_csv(scgpt_submission_path, index=False)
print("Saved:", scgpt_submission_path)

