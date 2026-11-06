"""
Nehasri_task2_combined.py
Task 2: Combined PCA + scGPT clustering pipeline (final strategy, ~88%+ on leaderboard).
Extracted from the submitted notebook. Requires scgpt_embeddings, geneformer_embeddings,
X_all_pca, and the Task 1.1/1.2 label arrays to already exist in memory -- run
Nehasri_task1_1_pca.py, Nehasri_task1_2_scgpt.py, and Nehasri_task1_2_geneformer.py
first, in that order, in the same session. (Geneformer's labels are used only for the
pairwise comparison printout below -- NOT in the Task 2 combination itself; see the
report for why Geneformer was excluded from Task 2's final pipeline.)
Produces Nehasri Eragandula_task2_submission.csv.

NOTE on plot title: the UMAP title string in the code below still reads
"Task 2 - Combined PCA+scGPT+Geneformer Clusters" from an earlier notebook revision.
This is a stale label only -- the actual combination (two cells above the plot) uses
combined = np.concatenate([pca_block_r, scgpt_block_r], axis=1), i.e. PCA + scGPT
only (60 total dimensions), with no Geneformer term. See the report for details.
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


pca_labels_full   = final_labels_pca_full.astype(int)
scgpt_labels_full = adata_scgpt_emb.obs["leiden_scgpt_final"].astype(int).to_numpy()
gf_labels_full    = adata_geneformer.obs["leiden_geneformer_final"].astype(int).to_numpy()

pairs = [
    ("PCA vs scGPT", pca_labels_full, scgpt_labels_full),
    ("PCA vs Geneformer", pca_labels_full, gf_labels_full),
    ("scGPT vs Geneformer", scgpt_labels_full, gf_labels_full),
]

comparison_rows = []
for name, a, b in pairs:
    comparison_rows.append({
        "comparison": name,
        "ARI": adjusted_rand_score(a, b),
        "NMI": normalized_mutual_info_score(a, b),
    })
comparison_df = pd.DataFrame(comparison_rows)

silhouette_summary = pd.DataFrame({
    "representation": ["PCA", "scGPT", "Geneformer"],
    "n_clusters": [len(np.unique(pca_labels_full)), len(np.unique(scgpt_labels_full)), len(np.unique(gf_labels_full))],
    "silhouette": [pca_final_silhouette, scgpt_silhouette, geneformer_silhouette],
})

print("========== PAIRWISE ARI / NMI ==========")
print(comparison_df.to_string(index=False))
print("\n========== SILHOUETTE SUMMARY (label-free) ==========")
print(silhouette_summary.to_string(index=False))

comparison_df.to_csv(os.path.join(ASSIGNMENT_DIR, f"{NAME}_task1_2_pairwise_comparison.csv"), index=False)
silhouette_summary.to_csv(os.path.join(ASSIGNMENT_DIR, f"{NAME}_task1_2_silhouette_summary.csv"), index=False)


from sklearn.decomposition import PCA as SKPCA

N_DIMS_PCA = 15
N_DIMS_SCGPT = 45

def reduce_block(X, n_dims):
    n_dims = min(n_dims, X.shape[1])
    return SKPCA(n_components=n_dims, random_state=42).fit_transform(X)

pca_block   = StandardScaler().fit_transform(X_all_pca)
scgpt_block = StandardScaler().fit_transform(scgpt_embeddings)

pca_block_r   = reduce_block(pca_block, N_DIMS_PCA)
scgpt_block_r = reduce_block(scgpt_block, N_DIMS_SCGPT)

combined = np.concatenate([pca_block_r, scgpt_block_r], axis=1)

print("Combined shape:", combined.shape)

# ---- Reduce combined space, then cluster ----
from sklearn.decomposition import PCA as SKPCA

n_reduce = min(50, combined.shape[1] - 1)
combined_reduced = SKPCA(n_components=n_reduce, random_state=42).fit_transform(combined)
print("Reduced combined shape:", combined_reduced.shape)

adata_combined = sc.AnnData(X=combined_reduced)
adata_combined.obs_names = adata.obs_names.astype(str)

sc.pp.neighbors(adata_combined, n_neighbors=15, use_rep="X", metric="euclidean")

resolutions = [0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50, 0.60, 0.80, 1.00, 1.20]
combined_res_results = []
for res in resolutions:
    key = f"leiden_{res}"
    sc.tl.leiden(adata_combined, resolution=res, key_added=key, random_state=42, flavor="igraph", directed=False)
    labels = adata_combined.obs[key].astype(str).to_numpy()
    n_clusters = len(np.unique(labels))
    sil = silhouette_score(combined_reduced, labels, metric="euclidean") if n_clusters > 1 else np.nan
    combined_res_results.append({"resolution": res, "n_clusters": n_clusters, "silhouette": sil})

combined_resolution_df = pd.DataFrame(combined_res_results)
print(combined_resolution_df.to_string(index=False))

best_combined_res = combined_resolution_df.loc[combined_resolution_df["silhouette"].idxmax(), "resolution"]
sc.tl.leiden(adata_combined, resolution=float(best_combined_res), key_added="leiden_combined_final",
             random_state=42, flavor="igraph", directed=False)
adata_combined.obs["leiden_combined_final"] = adata_combined.obs["leiden_combined_final"].astype(str)

combined_silhouette = silhouette_score(
    combined_reduced, adata_combined.obs["leiden_combined_final"].to_numpy(), metric="euclidean"
)
print("\nFinal combined resolution:", best_combined_res,
      "| clusters:", adata_combined.obs["leiden_combined_final"].nunique(),
      "| silhouette:", round(combined_silhouette, 4))


# ---- Combined UMAP + Task 2 submission ----
sc.tl.umap(adata_combined, min_dist=0.5, spread=1.0, random_state=42)
sc.pl.umap(adata_combined, color="leiden_combined_final", title="Task 2 - Combined PCA+scGPT+Geneformer Clusters",
           legend_loc="right margin", frameon=False, show=True)

combined_labels_1based = adata_combined.obs["leiden_combined_final"].astype(int).to_numpy() + 1
task2_submission = pd.DataFrame({"Id": np.arange(adata.n_obs), "Label": combined_labels_1based})
task2_submission_path = os.path.join(ASSIGNMENT_DIR, f"{NAME}_task2_submission.csv")
task2_submission.to_csv(task2_submission_path, index=False)
print("Saved:", task2_submission_path)
print(task2_submission.head())

