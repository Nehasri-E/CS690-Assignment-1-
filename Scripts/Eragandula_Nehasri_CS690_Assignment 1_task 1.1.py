"""
Nehasri_task1_1_pca.py
Task 1.1: PCA-based Leiden clustering pipeline.
Extracted from the submitted notebook (Eragandula_Nehasri_CS690_Assignment_1.ipynb).
Run top-to-bottom to reproduce Nehasri Eragandula_task1_1_submission.csv.
Requires dataset.h5ad and sample_submission.csv in ASSIGNMENT_DIR.
If not running in Colab, mount/set ASSIGNMENT_DIR to a local path with those files.
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


# ---- QC ----
sc.pp.calculate_qc_metrics(adata, inplace=True)

MIN_GENES, MAX_GENES = 900, 4000
qc_mask = (
    (adata.obs["n_genes_by_counts"] >= MIN_GENES) &
    (adata.obs["n_genes_by_counts"] <= MAX_GENES)
).to_numpy()

adata_qc = adata[qc_mask].copy()
print(f"QC: kept {adata_qc.n_obs} / {adata.n_obs} cells "
      f"({100*adata_qc.n_obs/adata.n_obs:.1f}%)")


# ---- Normalization check -----

adata_preprocessed = adata_qc.copy()
print("X range:", float(adata_preprocessed.X.min()), "to", float(adata_preprocessed.X.max()))


# ---- HVG + scale + PCA ----
sc.pp.highly_variable_genes(adata_preprocessed, n_top_genes=2000, flavor="seurat")
adata_hvg = adata_preprocessed[:, adata_preprocessed.var["highly_variable"]].copy()

adata_pca = adata_hvg.copy()
sc.pp.scale(adata_pca, zero_center=True, max_value=10)
sc.tl.pca(adata_pca, n_comps=50, svd_solver="arpack")

print("HVGs:", adata_hvg.n_vars)
print("PCA shape:", adata_pca.obsm["X_pca"].shape)
print("Variance explained (50 PCs):",
      f"{adata_pca.uns['pca']['variance_ratio'].sum()*100:.2f}%")


# ---- Neighbors + Leiden resolution sweep ----
sc.pp.neighbors(adata_pca, n_neighbors=15, n_pcs=50, use_rep="X_pca", metric="euclidean")

resolutions = [0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50, 0.60, 0.80, 1.00, 1.20, 1.50, 2.00]
results = []
for res in resolutions:
    key = f"leiden_{res}"
    sc.tl.leiden(adata_pca, resolution=res, key_added=key, random_state=42, flavor="igraph", directed=False)
    labels = adata_pca.obs[key].astype(str).to_numpy()
    n_clusters = len(np.unique(labels))
    sil = silhouette_score(adata_pca.obsm["X_pca"], labels, metric="euclidean") if n_clusters > 1 else np.nan
    results.append({"resolution": res, "n_clusters": n_clusters, "silhouette": sil})

pca_resolution_df = pd.DataFrame(results)
print(pca_resolution_df.to_string(index=False))

best_row = pca_resolution_df.loc[pca_resolution_df["silhouette"].idxmax()]
best_resolution_pca = best_row["resolution"]
print("\nBest resolution (by silhouette):", best_resolution_pca,
      "| clusters:", int(best_row["n_clusters"]),
      "| silhouette:", round(best_row["silhouette"], 4))


# ---- Final Leiden clustering at chosen resolution ----
FINAL_RESOLUTION_PCA = float(best_resolution_pca)

sc.tl.leiden(adata_pca, resolution=FINAL_RESOLUTION_PCA, key_added="leiden_final",
             random_state=42, flavor="igraph", directed=False)
adata_pca.obs["leiden_final"] = adata_pca.obs["leiden_final"].astype(str)

pca_final_silhouette = silhouette_score(
    adata_pca.obsm["X_pca"], adata_pca.obs["leiden_final"].to_numpy(), metric="euclidean"
)
print("Final resolution:", FINAL_RESOLUTION_PCA)
print("Number of clusters:", adata_pca.obs["leiden_final"].nunique())
print("Silhouette:", round(pca_final_silhouette, 4))
print("\nCluster sizes:")
print(adata_pca.obs["leiden_final"].value_counts().sort_index())


# ---- Differential expression / marker genes  ----
adata_de = adata_qc.copy()
adata_de.obs["leiden_final"] = adata_pca.obs["leiden_final"].to_numpy()

sc.tl.rank_genes_groups(adata_de, groupby="leiden_final", method="wilcoxon", key_added="rank_genes_final")

result = adata_de.uns["rank_genes_final"]
marker_records = []
for cluster in result["names"].dtype.names:
    genes = result["names"][cluster][:10]
    for rank, gene in enumerate(genes, start=1):
        marker_records.append({"cluster": cluster, "rank": rank, "gene": gene})
marker_df = pd.DataFrame(marker_records)

for cluster in result["names"].dtype.names:
    print(f"\nCluster {cluster}:")
    print(marker_df[marker_df["cluster"] == cluster][["rank", "gene"]].to_string(index=False))


# ---- Biological annotation ----
# Assigned from the marker-gene table above (classic PBMC/bone-marrow markers).
cluster_annotations = {
    "0":  "Naive T cells",                                  # ribosomal-high signature (LDHB, RPS29, RPS15A...)
    "1":  "Plasmacytoid Dendritic Cells (pDCs)",             # PLD4, CD74, HLA-DPA1, GSN
    "2":  "Classical Monocytes (CD14+)",                     # LYZ, S100A9, S100A8, FCN1
    "3":  "Natural Killer (NK) cells",                       # GNLY, NKG7, PRF1, KLRD1
    "4":  "B cells",                                         # CD79A, MS4A1, CD79B, BANK1
    "5":  "CD8+ Memory/Effector T cells",                    # CCL5, NKG7, GZMA, GZMK
    "6":  "Non-classical Monocytes (CD16+)",                 # LST1, AIF1, FCER1G, MS4A7
    "7":  "Platelets / Megakaryocytes",                      # PF4, PPBP, NRGN, GNG11
    "8":  "B cells (small subset)",                          # MS4A1, CD79A, RALGPS2
    "9":  "Hematopoietic Stem/Progenitor Cells (HSPCs)",     # CD34, SOX4, CYTL1, PRSS57
    "10": "Plasma cells",                                    # SEC11C, MZB1, SSR4, PDIA4
    "11": "Conventional Dendritic Cells (cDC1)",             # CLEC9A, BATF3, IDO1
}

adata_pca.obs["cell_type"] = adata_pca.obs["leiden_final"].map(cluster_annotations)
print(adata_pca.obs[["leiden_final", "cell_type"]].drop_duplicates().sort_values("leiden_final").to_string(index=False))


# ---- UMAP + plots ----
sc.tl.umap(adata_pca, min_dist=0.5, spread=1.0, random_state=42)

sc.pl.umap(adata_pca, color="leiden_final", title="Task 1.1 - PCA + Leiden Clusters",
           legend_loc="right margin", frameon=False, show=True)
sc.pl.umap(adata_pca, color="cell_type", title="Task 1.1 - Biological Annotations",
           legend_loc="right margin", frameon=False, show=True)


# ---- Back-fill labels for QC-dropped cells + build full-length submission ----
# The Kaggle submission needs ALL 10,727 original cells, but adata_pca only has the
# QC-passed subset. We train a KNN on the QC-passed PCA space and predict labels for
# the QC-dropped cells, using the SAME HVGs and the SAME scale (mean/std) as the final PCA.

hvg_genes = adata_hvg.var_names
adata_all_hvg = adata[:, hvg_genes].copy()
X_all = np.asarray(adata_all_hvg.X.todense() if hasattr(adata_all_hvg.X, "todense") else adata_all_hvg.X, dtype=np.float32)

scaler_mean = adata_pca.var["mean"].to_numpy()
scaler_std  = adata_pca.var["std"].to_numpy()
X_all_scaled = (X_all - scaler_mean) / scaler_std
X_all_scaled = np.clip(X_all_scaled, -10, 10)   # matches max_value=10 used in sc.pp.scale

# project into the same PCA space
pca_components = adata_pca.varm["PCs"]     # (n_hvg, 50)
X_all_pca = X_all_scaled @ pca_components

qc_labels = adata_pca.obs["leiden_final"].astype(str).to_numpy()
knn = KNeighborsClassifier(n_neighbors=15, metric="euclidean")
knn.fit(adata_pca.obsm["X_pca"], qc_labels)
all_predicted_labels = knn.predict(X_all_pca)

final_labels_pca_full = np.empty(adata.n_obs, dtype=object)
final_labels_pca_full[qc_mask]  = qc_labels
final_labels_pca_full[~qc_mask] = all_predicted_labels[~qc_mask]

print("QC-passed cells kept exact Leiden label:", qc_mask.sum())
print("QC-dropped cells filled via KNN:", (~qc_mask).sum())
print("Total labeled:", len(final_labels_pca_full), "| Unique clusters:", len(np.unique(final_labels_pca_full)))


# ---- Save Task 1.1 submission ----
sample_submission = pd.read_csv(SAMPLE_SUB_PATH) if os.path.exists(SAMPLE_SUB_PATH) else pd.DataFrame({"ID": np.arange(adata.n_obs)})

pca_submission = pd.DataFrame({
    "ID": sample_submission["ID"].to_numpy() if "ID" in sample_submission.columns else np.arange(adata.n_obs),
    "Label": final_labels_pca_full.astype(int) + 1,   # 1-based
})

pca_submission_path = os.path.join(ASSIGNMENT_DIR, f"{NAME}_task1_1_submission.csv")
pca_submission.to_csv(pca_submission_path, index=False)
print("Saved:", pca_submission_path)
print(pca_submission.head())

