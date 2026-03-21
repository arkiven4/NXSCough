import os, json, math, random, pickle, shutil, socket, inspect, argparse, subprocess, warnings
from collections import Counter
from transformers import AutoFeatureExtractor, ASTForAudioClassification
import gc

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import cm

import librosa
from tqdm import tqdm
from IPython.display import Audio

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
from torchvision import datasets, transforms

import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping
from lightning.pytorch.loggers.tensorboard import TensorBoardLogger

import umap
from sklearn.manifold import TSNE

from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA

from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, IsolationForest
from sklearn.mixture import GaussianMixture

from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.neighbors import KernelDensity, NearestNeighbors
from sklearn.svm import OneClassSVM
from sklearn.neighbors import LocalOutlierFactor, NearestNeighbors

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score

from sklearn.model_selection import StratifiedKFold, StratifiedGroupKFold, GroupKFold, train_test_split
from sklearn.neural_network import MLPClassifier

from sklearn.metrics import (
    roc_auc_score, roc_curve, average_precision_score,
    precision_recall_curve, classification_report,
    confusion_matrix, ConfusionMatrixDisplay,
    accuracy_score, balanced_accuracy_score,
    silhouette_score, silhouette_samples, davies_bouldin_score
)

from scipy import stats as scipy_stats
from diptest import diptest as _diptest

import train
import commons
import models
import utils
import lightning_wrapper

from cough_datasets import (
    CoughDatasets,
    CoughDatasetsCollate,
    CoughDiseaseBinaryBatchSampler
)

torch.set_float32_matmul_precision("medium")
cmap = cm.get_cmap("viridis")

colors = {"TB": "#C0392B", "Non-TB": "#2980B9"}
bins = np.linspace(0, 1, 30)

# %%
def _aprior(delta_hat: np.ndarray) -> np.ndarray:
    """ComBat EM prior: shape parameter for inverse gamma."""
    m = delta_hat.mean(axis=1)
    s2 = delta_hat.var(axis=1) + 1e-8
    return (2 * s2 + m**2) / s2

def _bprior(delta_hat: np.ndarray) -> np.ndarray:
    """ComBat EM prior: scale parameter for inverse gamma."""
    m = delta_hat.mean(axis=1)
    s2 = delta_hat.var(axis=1) + 1e-8
    return (m * s2 + m**3) / s2

def combat_harmonize(embeddings: np.ndarray,
    db_labels: np.ndarray, covariates: np.ndarray = None,
    n_iter: int = 100, conv_threshold: float = 1e-4) -> np.ndarray:
    """
    Empirical Bayes ComBat harmonization.
    
    Removes additive (gamma) and multiplicative (delta) batch effects
    while preserving biological signal via covariate design matrix.
    
    Parameters
    ----------
    embeddings   : (N, D) float array
    db_labels    : (N,) array of database IDs
    covariates   : (N, C) optional biological covariates to preserve
                   (e.g. disease_status encoded as 0/1)
    n_iter       : EM iterations for Bayesian shrinkage
    conv_threshold: convergence tolerance
    
    Returns
    -------
    harmonized   : (N, D) float array
    """
    N, D = embeddings.shape
    batches = np.unique(db_labels)
    n_batches = len(batches)
    
    # 1. Build design matrix
    # Intercept + covariates (biological signal to preserve)
    if covariates is not None:
        if covariates.ndim == 1:
            covariates = covariates.reshape(-1, 1)
        design = np.hstack([np.ones((N, 1)), covariates])
    else:
        design = np.ones((N, 1))
    
    # 2. Standardize overall
    grand_mean = embeddings.mean(axis=0)
    var_pooled = embeddings.var(axis=0) + 1e-8
    
    # 3. Regress out covariates to get residuals
    # Solve: embeddings = design @ B + residuals
    B, _, _, _ = np.linalg.lstsq(design, embeddings, rcond=None)
    residuals = embeddings - design @ B
    
    # 4. Standardize residuals
    stand_mean = (grand_mean / np.sqrt(var_pooled))
    s_data = (embeddings - grand_mean) / np.sqrt(var_pooled)
    
    # 5. Estimate batch effects (gamma_hat = additive, delta_hat = multiplicative)
    gamma_hat = np.zeros((n_batches, D))
    delta_hat = np.zeros((n_batches, D))
    
    batch_idx = {}
    for i, b in enumerate(batches):
        mask = db_labels == b
        batch_idx[i] = mask
        batch_data = s_data[mask]
        gamma_hat[i] = batch_data.mean(axis=0)
        delta_hat[i] = batch_data.var(axis=0) + 1e-8
    
    # 6. Empirical Bayes priors
    gamma_bar = gamma_hat.mean(axis=0)
    t2 = gamma_hat.var(axis=0) + 1e-8
    
    a_prior = _aprior(delta_hat)
    b_prior = _bprior(delta_hat)
    
    # 7. EM iteration for posterior estimates
    gamma_star = gamma_hat.copy()
    delta_star = delta_hat.copy()
    
    for _ in range(n_iter):
        gamma_star_new = np.zeros_like(gamma_hat)
        delta_star_new = np.zeros_like(delta_hat)
        
        for i, b in enumerate(batches):
            mask = batch_idx[i]
            n_i = mask.sum()
            batch_data = s_data[mask]
            
            # Posterior gamma (additive)
            gamma_star_new[i] = (
                (t2 * n_i * batch_data.mean(axis=0) + delta_star[i] * gamma_bar)
                / (t2 * n_i + delta_star[i])
            )
            
            # Posterior delta (multiplicative) via inverse gamma
            sum_sq = ((batch_data - gamma_star_new[i]) ** 2).sum(axis=0)
            delta_star_new[i] = (b_prior[i] + 0.5 * sum_sq) / (a_prior[i] + n_i / 2.0 - 1)
            delta_star_new[i] = np.maximum(delta_star_new[i], 1e-8)
        
        # Check convergence
        g_change = np.abs(gamma_star_new - gamma_star).max()
        d_change = np.abs(delta_star_new - delta_star).max()
        gamma_star = gamma_star_new
        delta_star = delta_star_new
        
        if g_change < conv_threshold and d_change < conv_threshold:
            break
    
    # 8. Apply correction
    harmonized = s_data.copy()
    for i, b in enumerate(batches):
        mask = batch_idx[i]
        harmonized[mask] = (
            (s_data[mask] - gamma_star[i]) / np.sqrt(delta_star[i])
        )
    
    # 9. Rescale back to original space
    harmonized = harmonized * np.sqrt(var_pooled) + grand_mean
    
    # Re-apply biological covariates
    harmonized = harmonized + design @ B - grand_mean
    
    return harmonized

def compute_mmd(X: np.ndarray, Y: np.ndarray, gamma: float = None) -> float:
    """
    Unbiased MMD² with RBF kernel.
    Lower = more similar distributions (better harmonization).
    """
    if gamma is None:
        gamma = 1.0 / X.shape[1]
    
    def rbf_kernel(A, B):
        dists = np.sum((A[:, None] - B[None, :]) ** 2, axis=2)
        return np.exp(-gamma * dists)
    
    Kxx = rbf_kernel(X, X)
    Kyy = rbf_kernel(Y, Y)
    Kxy = rbf_kernel(X, Y)
    
    n, m = len(X), len(Y)
    mmd2 = (
        (Kxx.sum() - np.trace(Kxx)) / (n * (n - 1))
        + (Kyy.sum() - np.trace(Kyy)) / (m * (m - 1))
        - 2 * Kxy.mean()
    )
    return float(mmd2)

def pairwise_mmd_matrix(embeddings: np.ndarray, db_labels: np.ndarray) -> pd.DataFrame:
    """Compute pairwise MMD between all database pairs."""
    batches = np.unique(db_labels)
    n = len(batches)
    matrix = np.zeros((n, n))
    
    for i, b1 in enumerate(batches):
        for j, b2 in enumerate(batches):
            if i != j:
                X = embeddings[db_labels == b1]
                Y = embeddings[db_labels == b2]
                # Subsample for speed if large
                if len(X) > 200: X = X[np.random.choice(len(X), 200, replace=False)]
                if len(Y) > 200: Y = Y[np.random.choice(len(Y), 200, replace=False)]
                matrix[i, j] = compute_mmd(X, Y)
    
    return pd.DataFrame(matrix, index=batches, columns=batches)

def get_classifiers() -> dict:
    """
    Return a dict of classifiers to compare.
    All wrapped in a Pipeline with StandardScaler + PCA for fair comparison.
    PCA(50) reduces 4096-dim embeddings to a stable space for all classifiers.
    """
    pca_step = ("pca", PCA(n_components=200, random_state=42))
    scale_step = ("scaler", StandardScaler())

    return {
        # "Logistic Regression": Pipeline([
        #     scale_step, pca_step,
        #     ("clf", LogisticRegression(max_iter=1000, C=1.0, random_state=42)),
        # ]),
        "Neural Network (MLP)": Pipeline([
            scale_step,
            pca_step,
            ("clf", MLPClassifier(
                hidden_layer_sizes=(128, 64),
                activation="relu",
                solver="adam",
                alpha=1e-4,
                batch_size=64,
                learning_rate_init=1e-3,
                max_iter=300,
                early_stopping=True,
                random_state=42
            )),
        ]),
        # "SVM (RBF)": Pipeline([
        #     scale_step, pca_step,
        #     ("clf", SVC(kernel="rbf", probability=True, C=1.0, random_state=42)),
        # ]),
        # "Random Forest": Pipeline([
        #     scale_step, pca_step,
        #     ("clf", RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)),
        # ]),
        # "Gradient Boosting": Pipeline([
        #     scale_step, pca_step,
        #     ("clf", GradientBoostingClassifier(n_estimators=200, random_state=42)),
        # ]),
    }

def compute_auc(y_true, y_proba):
    classes = np.unique(y_true)

    # Binary case
    if len(classes) == 2:
        # handle (n_samples,) or (n_samples, 2)
        if y_proba.ndim == 2:
            y_score = y_proba[:, 1]
        else:
            y_score = y_proba
        return roc_auc_score(y_true, y_score)

    # Multiclass case
    else:
        return roc_auc_score(
            y_true,
            y_proba,
            multi_class="ovr",
            average="macro"
        )

def run_embedding_classification(
    df,
    train_db,
    test_dbs=None,
    embed_col="embed",
    label_col="disease_status",
    participant_col="participant",
    db_col="db",
    n_splits=5,
    classifiers=None
):
    """
    Flexible evaluation pipeline for embedding-based classification.

    Parameters
    ----------
    df : DataFrame
        Combined dataframe containing embeddings and metadata.
    train_db : int or list
        DB id(s) used for training + cross-validation.
    test_dbs : list or None
        DB ids used as external evaluation sets.
    embed_col : str
        Column containing embeddings.
    label_col : str
        Label column.
    participant_col : str
        Column used for GroupKFold grouping.
    db_col : str
        Database identifier column.
    n_splits : int
        Number of CV folds.
    classifiers : dict
        Dict of sklearn classifiers.

    Returns
    -------
    results : dict
        Mean and std metrics across folds for each classifier.
    """

    if classifiers is None:
        classifiers = get_classifiers()

    if not isinstance(train_db, list):
        train_db = [train_db]

    if test_dbs is None:
        test_dbs = []

    df_train = df[df[db_col].isin(train_db)].copy()

    X = np.vstack(df_train[embed_col].values).astype(np.float32)
    y = df_train[label_col].astype(int).values
    groups = df_train[participant_col].values

    external_sets = {}
    for db in test_dbs:
        df_ext = df[df[db_col] == db]
        X_ext = np.vstack(df_ext[embed_col].values).astype(np.float32)
        y_ext = df_ext[label_col].astype(int).values
        external_sets[db] = (X_ext, y_ext)

    cv = StratifiedGroupKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=42
    )

    splits = list(cv.split(X, y, groups=groups))

    model_results = {}
    for clf_name, clf in tqdm(classifiers.items(), desc="Training classifiers"):
        fold_aurocs = []
        fold_aurocs_ext = {db: [] for db in test_dbs}

        for train_idx, test_idx in splits:
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]

            clf.fit(X_train, y_train)

            y_proba = clf.predict_proba(X_test)
            fold_aurocs.append(compute_auc(y_test, y_proba))

            for db in test_dbs:
                X_ext, y_ext = external_sets[db]
                y_proba_ext = clf.predict_proba(X_ext)
                fold_aurocs_ext[db].append(compute_auc(y_ext, y_proba_ext))

        model_results[clf_name] = {
            "auroc": np.mean(fold_aurocs),
            **{
                f"auroc_db{db}": np.mean(vals)
                for db, vals in fold_aurocs_ext.items()
            }
        }
        
    # -------- aggregate across models --------
    all_aurocs = [v["auroc"] for v in model_results.values()]

    summary = {
        "all": f"{np.mean(all_aurocs):.2f} ± {np.std(all_aurocs):.2f}"
    }

    for db in test_dbs:
        vals = [v[f"auroc_db{db}"] for v in model_results.values()]
        summary[f"db_{db}"] = f"{np.mean(vals):.2f} ± {np.std(vals):.2f}"

    return summary, model_results

def _sigmoid_normalize(scores: np.ndarray) -> np.ndarray:
    """Center and sigmoid-normalize scores to [0, 1]."""
    centered = (scores - scores.mean()) / (scores.std() + 1e-8)
    return 1.0 / (1.0 + np.exp(-centered))

def _minmax_normalize(scores: np.ndarray) -> np.ndarray:
    lo, hi = scores.min(), scores.max()
    return (scores - lo) / (hi - lo + 1e-8)

def best_gmm(data, max_k=6, random_state=42):
    best_bic, best_k = np.inf, 1
    max_k = min(max_k, len(data) // 20)  # need enough samples per component
    max_k = max(max_k, 1)
    for k in tqdm(range(1, max_k + 1)):
        try:
            g = GaussianMixture(n_components=k, covariance_type="diag", random_state=random_state, n_init=3)
            g.fit(data)
            bic = g.bic(data)
            if bic < best_bic:
                best_bic, best_k = bic, k
        except Exception:
            break
    gmm = GaussianMixture(n_components=best_k, covariance_type="diag",
                            random_state=random_state, n_init=5)
    gmm.fit(data)
    print(f"    GMM components selected (BIC): {best_k}")
    return gmm

def find_optimal_k(
    embeddings: np.ndarray,
    k_range: range = None,
    random_state: int = 42,
) -> tuple[int, pd.DataFrame]:
    """
    Find optimal number of clusters via silhouette + elbow analysis.
    
    Returns
    -------
    best_k     : recommended number of clusters
    metrics_df : per-k metrics table
    """
    if k_range is None:
        k_range = range(2, min(8, len(embeddings) // 10 + 1))
    
    pca = PCA(n_components=min(50, embeddings.shape[1]), random_state=random_state)
    emb_pca = pca.fit_transform(embeddings)
    
    records = []
    for k in k_range:
        km = KMeans(n_clusters=k, n_init=10, random_state=random_state)
        labels = km.fit_predict(emb_pca)
        
        sil = silhouette_score(emb_pca, labels)
        inertia = km.inertia_
        db = davies_bouldin_score(emb_pca, labels)
        records.append({"k": k, "silhouette": sil, "inertia": inertia, "davies_bouldin": db})
    
    df = pd.DataFrame(records)
    
    # Best k = highest silhouette (or use elbow)
    best_k = int(df.loc[df["silhouette"].idxmax(), "k"])
    
    return best_k, df

def gmm_acoustic_threshold(acoustic_scores, tb_mask, max_k=6, random_state=42):
    scores = np.asarray(acoustic_scores)
    scores_2d = scores.reshape(-1, 1)

    # ── BIC model selection ─────────────────────────
    bics, models = [], []
    for k in range(1, max_k + 1):
        try:
            g = GaussianMixture(
                n_components=k,
                covariance_type="full",
                n_init=10,
                random_state=random_state,
                max_iter=300
            )
            g.fit(scores_2d)
            bics.append(g.bic(scores_2d))
            models.append(g)
        except Exception:
            bics.append(np.inf)
            models.append(None)

    best_k = int(np.argmin(bics)) + 1
    best_gmm = models[best_k - 1]

    # ── fallback if only 1 component ────────────────
    if best_k == 1:
        print(f" Return to Percentile Threshold")
        return float(np.percentile(scores[~tb_mask], 95))

    # ── posterior crossover threshold ───────────────
    means = best_gmm.means_.flatten()
    order = np.argsort(means)

    comp_acoustic = int(order[-1])

    x_range = np.linspace(scores.min(), scores.max(), 10000).reshape(-1, 1)
    p_range = best_gmm.predict_proba(x_range)[:, comp_acoustic]

    cross_idx = int(np.argmin(np.abs(p_range - 0.5)))
    final_threshold = float(x_range[cross_idx, 0])

    return final_threshold

def majority_vote(group):
    counts = group["disease_status_rev"].value_counts(normalize=True)
    if counts.get(1, 0) > 0.51:
        return 1
    elif counts.get(2, 0) > 0.51:
        return 2
    else:
        return 2

def print_tb_stats(df_temp, tb_mask0):
    tb_df = df_temp[tb_mask0]

    vc = tb_df["disease_status_rev"].value_counts()
    total = len(tb_df)

    print(f"Acoustic-TB: {vc.get(1,0)} ({vc.get(1,0)/total*100:.1f}%) | "
          f"Label-only-TB: {vc.get(2,0)} ({vc.get(2,0)/total*100:.1f}%)")

    db_counts = tb_df.groupby(["db","disease_status_rev"]).size().unstack(fill_value=0)

    print("Acoustic-TB -> " + " | ".join(
        f"DB {db}: {int(db_counts.loc[db].get(1,0))} ({v:.1f}%)"
        for db, v in (db_counts.get(1, 0).div(db_counts.sum(1))*100).items()
    ) + "\n")

def revised_label_df(df_combine, acoustic_scores, global_thres=True, majority_participant=True):
    df_temp = df_combine.copy()
    df_temp["acoustic_tb_score"] = acoustic_scores

    tb_mask0 = df_temp["disease_status"] == 1
    df_temp["disease_status_rev"] = 0

    #threshold_0 = df_temp[tb_mask0]['acoustic_tb_score'].median()
    if global_thres:
        threshold_0 = gmm_acoustic_threshold(acoustic_scores, tb_mask0)
        df_temp.loc[tb_mask0, "disease_status_rev"] = np.where(
            df_temp.loc[tb_mask0, "acoustic_tb_score"].values >= threshold_0,
            1, 2
        )
    else:
        for db in df_temp["db"].unique():
            db_mask = df_temp["db"] == db
            mask = db_mask & tb_mask0

            scores_db = df_temp.loc[db_mask, "acoustic_tb_score"].values
            tb_mask_db = df_temp.loc[db_mask, "disease_status"].values == 1

            threshold_db = gmm_acoustic_threshold(scores_db, tb_mask_db)
            df_temp.loc[mask, "disease_status_rev"] = np.where(
                df_temp.loc[mask, "acoustic_tb_score"].values >= threshold_db,
                1, 2
            )

    if majority_participant:
        # Step 2: participant-level majority voting
        participant_labels = (
            df_temp.groupby("participant")
            .apply(majority_vote, include_groups=False)
            .rename("participant_label")
        )

        df_temp = df_temp.merge(
            participant_labels, left_on="participant", right_index=True, how="left"
        )
        
    print_tb_stats(df_temp, tb_mask0)
    return df_temp

def evaluate_acoustic_tb(df_combine, acoustic_scores, global_thres=True):
    df_temp = revised_label_df(df_combine, acoustic_scores, global_thres=global_thres)

    df_temp_eval = (
        df_temp[
            (df_temp["disease_status_rev"] != 2) & (df_temp["db"] != 3)
        ]
        .copy()
        .reset_index(drop=True)
    )

    summary, model_results = run_embedding_classification(
        df_temp_eval,
        train_db=0,
        test_dbs=[1, 2],
        label_col="disease_status_rev",
    )

    print(summary)
    return df_temp

def _auto_pca_components(
    embeddings: np.ndarray,
    variance_threshold: float = 0.95,
    random_state: int = 42,
) -> int:
    """
    Find minimum n_components that explains `variance_threshold` of total variance.
    Fits PCA with full components, reads cumulative explained variance ratio.

    Parameters
    ----------
    variance_threshold : float in (0, 1]
        Target cumulative explained variance. Default 0.95 (95%).
        Use 0.99 for higher fidelity, 0.90 for more compression.

    Returns
    -------
    n_components : int — minimum dims to reach the threshold
    explained    : float — actual variance explained at that n
    """
    max_comp = min(embeddings.shape[0] - 1, embeddings.shape[1])
    pca_full = PCA(n_components=max_comp, random_state=random_state)
    pca_full.fit(embeddings)
    cumvar = np.cumsum(pca_full.explained_variance_ratio_)
    n_comp = int(np.searchsorted(cumvar, variance_threshold) + 1)
    n_comp = min(n_comp, max_comp)
    explained = cumvar[n_comp - 1]
    print(f"      Auto PCA: {n_comp} components explain {explained*100:.1f}% variance "
          f"(threshold={variance_threshold*100:.0f}%)")
    return n_comp, explained

def _posterior(log_p_tb, log_p_nontb):
    """Numerically stable Bayesian posterior P(TB|x) from log-likelihoods."""
    log_max = np.maximum(log_p_tb, log_p_nontb)
    num = np.exp(log_p_tb - log_max)
    den = num + np.exp(log_p_nontb - log_max) + 1e-8
    return (num / den).astype(np.float32)

def clip_scores(scores, tb_mask=None, min_samples=50):
    scores = np.asarray(scores)
    scores_clipped = scores.copy()

    # --- Global fallback ---
    if tb_mask is None:
        Q1 = np.percentile(scores, 25)
        Q3 = np.percentile(scores, 75)
        IQR = Q3 - Q1

        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR

        return np.clip(scores, lower, upper)

    # --- Ensure boolean mask ---
    tb_mask = np.asarray(tb_mask).astype(bool)
    ntb_mask = ~tb_mask

    # --- TB group ---
    tb_scores = scores[tb_mask]
    if len(tb_scores) >= min_samples:
        Q1_tb = np.percentile(tb_scores, 25)
        Q3_tb = np.percentile(tb_scores, 75)
        IQR_tb = Q3_tb - Q1_tb

        lower_tb = Q1_tb - 1.5 * IQR_tb
        upper_tb = Q3_tb + 1.5 * IQR_tb

        scores_clipped[tb_mask] = np.clip(tb_scores, lower_tb, upper_tb)

    # --- Non-TB group ---
    ntb_scores = scores[ntb_mask]
    if len(ntb_scores) >= min_samples:
        Q1_ntb = np.percentile(ntb_scores, 25)
        Q3_ntb = np.percentile(ntb_scores, 75)
        IQR_ntb = Q3_ntb - Q1_ntb

        lower_ntb = Q1_ntb - 1.5 * IQR_ntb
        upper_ntb = Q3_ntb + 1.5 * IQR_ntb

        scores_clipped[ntb_mask] = np.clip(ntb_scores, lower_ntb, upper_ntb)

    return scores_clipped

def knn_tb_score(emb_reduced, tb_mask, automate_k=False, k_tb=15, k_nontb=15, alpha=20, metric="euclidean"):
    tb_emb   = emb_reduced[tb_mask]
    nontb_emb = emb_reduced[~tb_mask]

    kw_nn = {"metric": metric}
    if metric == "mahalanobis":
        cov = np.cov(emb_reduced.T)
        VI = np.linalg.inv(cov + np.eye(cov.shape[0]) * 1e-6)
        kw_nn["metric_params"] = {"VI": VI}

    if automate_k:
        k_tb, _ = find_optimal_k(tb_emb, random_state=42)
        k_nontb, _ = find_optimal_k(nontb_emb, random_state=42)

    print(f"k_tb: {k_tb} | k_nontb: {k_nontb}")

    # Fit separate NN indices for each class
    nn_tb = NearestNeighbors(n_neighbors=k_tb, n_jobs=-1, **kw_nn).fit(tb_emb)
    nn_nontb = NearestNeighbors(n_neighbors=k_nontb, n_jobs=-1, **kw_nn).fit(nontb_emb)

    # Query distances for all samples
    d_tb, _ = nn_tb.kneighbors(emb_reduced)
    d_nontb, _ = nn_nontb.kneighbors(emb_reduced)

    mean_d_tb = d_tb.mean(axis=1)
    mean_d_nontb = d_nontb.mean(axis=1)

    #mean_d_tb = clip_scores(mean_d_tb)
    #mean_d_nontb = clip_scores(mean_d_nontb)

    # Total Distance Score, [nonTB] ----------- x --- [TB] -> 0 - 1, tell how far from non tb
    # scores_raw = mean_d_nontb / (mean_d_tb + mean_d_nontb + 1e-8) # Relative proximity score: high = closer to TB
    margin = mean_d_nontb - mean_d_tb
    margin[tb_mask] = clip_scores(margin[tb_mask])
    margin[~tb_mask] = clip_scores(margin[~tb_mask])
    scores_raw = 1 / (1 + np.exp(-alpha * margin))  
    return _minmax_normalize(scores_raw)

def validate_acoustic_score_indiv(embeddings, disease_statuss, acoustic_scores, k_neighbours: int = 15, random_state: int = 42) -> dict:
    """
    Four principled tests that the acoustic_tb_score reflects a real
    acoustic TB signal rather than an arbitrary or coincidental ranking.

    All tests are label-agnostic in spirit — they do not use AUROC
    against TB/NonTB labels as a criterion, avoiding the circularity
    of evaluating a score built from those same labels.

    Tests
    ─────
    1. Bimodality (Hartigan's Dip Test)
       Tests whether the TB score distribution is bimodal — a necessary
       condition if Acoustic-TB and Label-only-TB are real subgroups.
       A unimodal distribution means the threshold is arbitrary.
       → p < 0.05 on TB scores supports the existence of two subgroups.

    5. Neighbourhood Purity
       In the embedding space: what fraction of each Acoustic-TB sample's
       k nearest neighbours are also Acoustic-TB vs Label-only-TB vs NonTB?
       If the score-derived labels are real, Acoustic-TB samples should
       cluster with each other more than with Label-only-TB or NonTB.
       This tests geometric consistency — a necessary condition for reality.
       → Acoustic-TB purity >> Label-only-TB purity = geometrically consistent.

    6. Permutation Test
       Randomly shuffle TB/NonTB labels 1000 times, recompute score
       separation (TB mean - NonTB mean) each time.
       If real label structure produces separation unlikely under random
       labelling, the score is detecting something real about TB labels.
       → p < 0.05 = real separation unlikely by chance.

    Returns
    -------
    dict with per-test results, overall verdict, and saved plots.
    """

    tb_mask = disease_statuss == 1
    tb_scores = acoustic_scores[tb_mask]
    nt_scores = acoustic_scores[~tb_mask]
    results   = {}

    # ── Test 1: Bimodality (Hartigan's Dip Test) ──────────────
    dip_stat, dip_pval = _diptest(tb_scores)
    bimodal = dip_pval < 0.05
    results["bimodality"] = {
        "dip_stat":  float(dip_stat),
        "dip_pval":  float(dip_pval),
        "bimodal":   bimodal,
        "n_tb":      int(tb_mask.sum()),
    }
    sig = "***" if dip_pval < 0.001 else "**" if dip_pval < 0.01 else "*" if dip_pval < 0.05 else "n.s."
    verdict1 = "✓ BIMODAL — two subgroups supported" if bimodal else "✗ UNIMODAL — threshold may be arbitrary"
    #print(f"  Dip statistic: {dip_stat:.4f}  p={dip_pval:.4f} {sig}")
    print(f"  → {verdict1}")

    # ── Test 5: Neighbourhood Purity ──────────────────────────
    # Reconstruct subtype labels using threshold from acoustic scores
    # Use 95th pct of NonTB as threshold (same as classify_tb_subgroups)
    threshold = np.percentile(nt_scores, 95)
    subtype = np.array(["Non-TB"] * len(embeddings), dtype=object)
    tb_idx  = np.where(tb_mask)[0]
    for i, idx in enumerate(tb_idx):
        subtype[idx] = "Acoustic-TB" if tb_scores[i] >= threshold else "Label-only-TB"

    from sklearn.neighbors import NearestNeighbors
    nn = NearestNeighbors(n_neighbors=k_neighbours + 1, metric="euclidean", n_jobs=-1)
    nn.fit(embeddings)
    _, indices = nn.kneighbors(embeddings)
    indices    = indices[:, 1:]   # exclude self

    purity = {}
    for target_type in ["Acoustic-TB", "Label-only-TB"]:
        target_idx = np.where(subtype == target_type)[0]
        if len(target_idx) == 0:
            continue

        same_frac, atb_frac, lonly_frac, ntb_frac = [], [], [], []
        for idx in target_idx:
            nbr_subtypes = subtype[indices[idx]]
            same_frac.append((nbr_subtypes == target_type).mean())
            atb_frac.append((nbr_subtypes == "Acoustic-TB").mean())
            lonly_frac.append((nbr_subtypes == "Label-only-TB").mean())
            ntb_frac.append((nbr_subtypes == "Non-TB").mean())

        purity[target_type] = {
            "n":            len(target_idx),
            "same_purity":  float(np.mean(same_frac)),
            "atb_frac":     float(np.mean(atb_frac)),
            "labelonly_frac": float(np.mean(lonly_frac)),
            "nontb_frac":   float(np.mean(ntb_frac)),
        }

        # print(f"  {target_type} (n={len(target_idx)}):")
        # print(f"    Neighbours that are same subtype : {np.mean(same_frac)*100:.1f}%")
        # print(f"    Neighbours that are Acoustic-TB  : {np.mean(atb_frac)*100:.1f}%")
        # print(f"    Neighbours that are Label-only-TB: {np.mean(lonly_frac)*100:.1f}%")
        # print(f"    Neighbours that are Non-TB       : {np.mean(ntb_frac)*100:.1f}%")

    results["neighbourhood_purity"] = purity

    # Purity verdict: Acoustic-TB should have higher same-type purity than Label-only-TB
    atb_purity   = purity.get("Acoustic-TB",    {}).get("same_purity", 0)
    lonly_purity = purity.get("Label-only-TB",  {}).get("same_purity", 0)
    lonly_nontb  = purity.get("Label-only-TB",  {}).get("nontb_frac", 0)
    consistent_purity = (atb_purity > lonly_purity) and (lonly_nontb > 0.2)
    verdict5 = (
        f"✓ GEOMETRICALLY CONSISTENT — Acoustic-TB clusters with itself "
        f"({atb_purity*100:.1f}%), Label-only-TB mixes with Non-TB ({lonly_nontb*100:.1f}%)"
        if consistent_purity else
        f"✗ NOT CONSISTENT — subtypes don't separate geometrically "
        f"(Acoustic-TB purity={atb_purity*100:.1f}%)"
    )
    print(f"  → {verdict5}")

    # ── Overall verdict ────────────────────────────────────────
    n_pass = sum([
        bimodal,
        consistent_purity,
    ])
    overall = (
        "✓ SCORE IS MEANINGFUL" if n_pass >= 2 else
        "⚠ SCORE IS PARTIALLY SUPPORTED" if n_pass == 1 else
        "✗ SCORE MAY NOT BE MEANINGFUL"
    )
    results["n_tests_passed"] = n_pass
    results["overall_verdict"] = overall

    # print("\n" + "─" * 60)
    # print(f"  Tests passed: {n_pass}/3")
    # print(f"  {overall}")
    # print("─" * 60)

    return results

def plot_tb_score_distributions(df_temp, bins, colors, figsize=(20, 10)):
    """
    Plot histogram + boxplot comparison of TB likelihood scores.

    Parameters
    ----------
    score_key : str
        Key to select which score to visualize (e.g., "knn_euc").
    bins : int or sequence
        Histogram bins.
    colors : dict
        Color mapping, e.g. {"TB": "...", "Non-TB": "..."}.
    masks : list of tuple
        [(mask_tb, label_tb, label_non_tb), ...]
    titles : list[str], optional
        Column titles.
    figsize : tuple
        Figure size.
    """
    
    sel_scores = df_temp['acoustic_tb_score'].values
    n_cols = len(df_temp['db'].unique()) + 1

    fig, axes = plt.subplots(2, n_cols, figsize=figsize)
    for i, (idx_db) in enumerate(range(n_cols)):
        idx_db = idx_db - 1
        if idx_db == -1:
            tb_mask = df_temp['disease_status'] == 1
            ntb_mask = df_temp['disease_status'] == 0
        else:
            tb_mask = (df_temp['disease_status'] == 1) & (df_temp['db'] == idx_db)
            ntb_mask = (df_temp['disease_status'] == 0) & (df_temp['db'] == idx_db)

        non_tb_scores = sel_scores[ntb_mask]
        tb_scores = sel_scores[tb_mask]
        dise_label = df_temp[ntb_mask | tb_mask]['disease_status'].values

        auc = roc_auc_score(dise_label, sel_scores[ntb_mask | tb_mask]) 
        non_tb_label, tb_label = (f"Non-TB DB{idx_db}", f"TB DB{idx_db}")

        # Histogram
        ax = axes[0, i]
        ax.hist(
            non_tb_scores,
            bins=bins,
            color=colors["Non-TB"],
            alpha=0.65,
            label=f"{non_tb_label} (μ={non_tb_scores.mean():.2f})",
            density=True
        )
        ax.hist(
            tb_scores,
            bins=bins,
            color=colors["TB"],
            alpha=0.65,
            label=f"{tb_label} (μ={tb_scores.mean():.2f})",
            density=True
        )

        ax.set_xlabel("Acoustic TB Likelihood Score")
        ax.set_ylabel("Density")
        ax.legend()
        ax.set_title(f"All TB vs Non-TB DB{idx_db} {auc:0.3f}")

        # Boxplot
        ax = axes[1, i]
        data_plot = [non_tb_scores, tb_scores]

        bp = ax.boxplot(
            data_plot,
            patch_artist=True,
            widths=0.5,
            medianprops=dict(color="white", linewidth=2)
        )

        for patch, color in zip(bp["boxes"], [colors["Non-TB"], colors["TB"]]):
            patch.set_facecolor(color)
            patch.set_alpha(0.75)

        ax.set_xticks([1, 2])
        ax.set_xticklabels([non_tb_label, tb_label], fontsize=10)
        ax.set_ylabel("Score", fontsize=9)

    plt.tight_layout()
    plt.show()

def plot_tb_split_filter(df_temp, bins, figsize=(16, 5)):
    """
    Plot histogram + boxplot comparison of TB likelihood scores.

    Parameters
    ----------
    score_key : str
        Key to select which score to visualize (e.g., "knn_euc").
    bins : int or sequence
        Histogram bins.
    colors : dict
        Color mapping, e.g. {"TB": "...", "Non-TB": "..."}.
    masks : list of tuple
        [(mask_tb, label_tb, label_non_tb), ...]
    titles : list[str], optional
        Column titles.
    figsize : tuple
        Figure size.
    """
    
    sel_scores = df_temp['acoustic_tb_score'].values
    n_cols = len(df_temp['db'].unique()) + 1

    fig, axes = plt.subplots(1, n_cols, figsize=figsize)
    for i, (idx_db) in enumerate(range(n_cols)):
        idx_db = idx_db - 1
        if idx_db == -1:
            tb_mask = df_temp['disease_status'] == 1
            ntb_mask = df_temp['disease_status'] == 0
        else:
            tb_mask = (df_temp['disease_status'] == 1) & (df_temp['db'] == idx_db)
            ntb_mask = (df_temp['disease_status'] == 0) & (df_temp['db'] == idx_db)

        non_tb_scores = sel_scores[ntb_mask]
        acoustic_tb_scores0 = df_temp[tb_mask].loc[df_temp["disease_status_rev"] == 1, "acoustic_tb_score"].values
        labelonly_tb_scores0 = df_temp[tb_mask].loc[df_temp["disease_status_rev"] == 2, "acoustic_tb_score"].values

        # Histogram
        ax = axes[i]
        ax.hist(non_tb_scores, bins=bins, color="#2980B9", alpha=0.6, label="Non-TB", density=True)
        ax.hist(acoustic_tb_scores0, bins=bins, color="#C0392B", alpha=0.7, label="Acoustic-TB", density=True)
        ax.hist(labelonly_tb_scores0, bins=bins, color="#F39C12", alpha=0.7, label="Label-only-TB", density=True)
        #ax.axvline(threshold_0, color="black", linestyle="--", linewidth=1.5, label=f"Threshold ({threshold_0:.2f})")

        ax.set_xlabel("Acoustic TB Likelihood Score")
        ax.set_ylabel("Density")
        ax.set_title("Score Distribution")
        ax.legend()

    plt.tight_layout()
    plt.show()

import torch
import torch.nn as nn
import torch.optim as optim

class AffineCouplingLayer(nn.Module):
    """Simple RealNVP affine coupling layer."""
    def __init__(self, dim, hidden):
        super().__init__()
        half = dim // 2
        self.net_s = nn.Sequential(
            nn.Linear(half, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
            nn.Linear(hidden, dim - half),
        )
        self.net_t = nn.Sequential(
            nn.Linear(half, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
            nn.Linear(hidden, dim - half),
        )
        self.half = half

    def forward(self, x):
        x1, x2 = x[:, :self.half], x[:, self.half:]
        s = self.net_s(x1).clamp(-2, 2)
        t = self.net_t(x1)
        y2 = x2 * s.exp() + t
        log_det = s.sum(dim=-1)
        return torch.cat([x1, y2], dim=-1), log_det

    def inverse(self, y):
        y1, y2 = y[:, :self.half], y[:, self.half:]
        s = self.net_s(y1).clamp(-2, 2)
        t = self.net_t(y1)
        x2 = (y2 - t) * (-s).exp()
        return torch.cat([y1, x2], dim=-1)

class RealNVP(nn.Module):
    def __init__(self, dim, n_flows, hidden):
        super().__init__()
        self.layers = nn.ModuleList([
            AffineCouplingLayer(dim, hidden) for _ in range(n_flows)
        ])

    def log_prob(self, x):
        log_det_sum = torch.zeros(x.shape[0], device=x.device)
        z = x
        for layer in self.layers:
            z, log_det = layer(z)
            log_det_sum += log_det
        # Standard Gaussian log-prob
        log_pz = -0.5 * (z ** 2 + np.log(2 * np.pi)).sum(dim=-1)
        return log_pz + log_det_sum

# %%
# from s3prl.upstream.mockingjay.builder import PretrainedTransformer
# from s3prl.upstream.mockingjay.model import TransformerSpecPredictionHead
# import s3prl.optimizers
# import sys
# original_optimizer = sys.modules.get("optimizers")
# sys.modules["optimizers"] = s3prl.optimizers

# class TERA_TryDownstream(nn.Module):
#     def __init__(self, input_size, **kwargs):
#         super(TERA_TryDownstream, self).__init__()

#         options = {
#             "load_pretrain": "True",
#             "no_grad": "True",
#             "dropout": "default",
#             "spec_aug": "False",
#             "spec_aug_prev": "False",
#             "output_hidden_states": "True",
#             "permute_input": "False",
#         }
#         options["ckpt_file"] = "/run/media/fourier/Data1/Pras/Thesis_Nexus/s3prl/s3prl/result/pretrain/tera_cough_ssldata_lowlr/states-990000.ckpt"
#         options["select_layer"] = -1
        
#         pretrained_dict = torch.load(options["ckpt_file"], weights_only=False)
#         transformer_state = pretrained_dict['Transformer']
#         spechead_state = pretrained_dict['SpecHead']

#         self.tera_model = PretrainedTransformer(options, inp_dim=-1)
#         self.tera_model.model.load_state_dict(transformer_state, strict=True)
#         self.tera_model.eval()

#         self.spechead_model = TransformerSpecPredictionHead(self.tera_model.model_config, self.tera_model.spec_dim)
#         self.spechead_model.load_state_dict(spechead_state, strict=True)
#         self.spechead_model.eval()

#     def forward(self, x, attention_mask=None, grl_lambda=0.0):
#         x = x.squeeze(1)
#         with torch.no_grad():
#             x = self.tera_model(x)[0] # Index 0 = Last Hidden, Index 1 All Transformwer
#             x = torch.nan_to_num(x, nan=0.0)
#             reconstructed_mel, _ = self.spechead_model(x)
#             reconstructed_mel = reconstructed_mel.transpose(1, 2)

#         mean = x.mean(dim=1)
#         std = x.std(dim=1)
#         feature_embedding = torch.cat([mean, std], dim=1)
#         return {
#             "embeddings": feature_embedding,
#             "reconstructed_mel": reconstructed_mel,
#         }
    
# model = TERA_TryDownstream(1)
# model.cuda()
# model.eval()
# del sys.modules["optimizers"]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
feature_extractor = AutoFeatureExtractor.from_pretrained("MIT/ast-finetuned-audioset-10-10-0.4593")
model_ast = ASTForAudioClassification.from_pretrained(
    "MIT/ast-finetuned-audioset-10-10-0.4593"
).to(device)
model_ast.eval()

def collate_fn_ast(batch_paths):
    audios = []
    for path in batch_paths:
        y, _ = librosa.load(path, sr=16000)
        audios.append(y)
    inputs = feature_extractor(audios, sampling_rate=16000, return_tensors="pt", padding=True)
    return inputs, batch_paths

def clean_by_ast(df_uncleaned, threshold=0.95, batch_size=32, num_workers=8):
    """
    Remove worst samples based on AST cough likelihood.

    Args:
        df_uncleaned : DataFrame with 'path_file'
        threshold    : quantile (e.g. 0.95 = remove worst 5%)

    Returns:
        df_clean, mask_remove
    """

    loader = DataLoader(
        df_uncleaned["path_file"].tolist(),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn_ast
    )

    results = []
    target_id = model_ast.config.label2id["Cough"]

    model_ast.eval()
    with torch.no_grad():
        for inputs, paths in tqdm(loader):
            inputs = {k: v.to(device) for k, v in inputs.items()}

            logits = model_ast(**inputs).logits
            log_probs = F.log_softmax(logits, dim=-1)
            losses = -log_probs[:, target_id]
            losses = losses.detach().cpu().numpy()

            results.extend(
                {"path_file": p, "loss": float(l)}
                for p, l in zip(paths, losses)
            )

    df_loss = pd.DataFrame(results)

    thr_value = df_loss["loss"].quantile(threshold)
    loss_map = df_loss.set_index("path_file")["loss"]
    mask_remove = df_uncleaned["path_file"].map(loss_map) >= thr_value

    df_clean = df_uncleaned[~mask_remove]
    total_removed = mask_remove.sum()
    total = len(df_uncleaned)

    # overall
    print(f"Removed: {total_removed} / {total} ({total_removed/total*100:.2f}%)")

    # class-wise
    for cls in [0, 1]:
        cls_mask = df_uncleaned["disease_status"] == cls
        cls_total = cls_mask.sum()
        cls_removed = (mask_remove & cls_mask).sum()

        pct_within_class = cls_removed / cls_total * 100 if cls_total > 0 else 0
        pct_of_removed = cls_removed / total_removed * 100 if total_removed > 0 else 0

        print(
            f"Class {cls} → Removed: {cls_removed}/{cls_total} "
            f"({pct_within_class:.2f}% of class, {pct_of_removed:.2f}% of removed)"
        )
        
    return df_clean

class Qwen3_Extractor(nn.Module):
    def __init__(self, dummy, **kwargs):
        super(Qwen3_Extractor, self).__init__()

        from transformers import Qwen3OmniMoeThinkerForConditionalGeneration
        temp_model = Qwen3OmniMoeThinkerForConditionalGeneration.from_pretrained(
            "/run/media/fourier/Data1/Pras/pretrain_models/Qwen3-Omni-30B-A3B-Thinking",
            torch_dtype="auto",
            device_map="cpu"
        )
        self.audio_tower = temp_model.audio_tower
        self.audio_tower.cuda()
        del temp_model
        import gc
        import torch
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        self.audiotower_hidden_dim = self.audio_tower.config.output_dim

    def after_cnn_len(self, L):
        L = (L - 1) // 2 + 1
        L = (L - 1) // 2 + 1
        L = (L - 1) // 2 + 1
        return L

    def forward(self, input_features, attention_mask=None, **kwargs):
        input_features = input_features.to(torch.bfloat16)
        feature_attention_mask = attention_mask.long()
        if feature_attention_mask is not None:
            audio_feature_lengths = torch.sum(feature_attention_mask, dim=1)
            input_features = input_features.permute(
                0, 2, 1)[feature_attention_mask.bool()].permute(1, 0)
        else:
            audio_feature_lengths = None

        feature_lens = audio_feature_lengths if audio_feature_lengths is not None else feature_attention_mask.sum(-1)
        audio_outputs = self.audio_tower(
            input_features,
            feature_lens=feature_lens,
        )
        audio_features = audio_outputs.last_hidden_state
        post_lens = torch.tensor(
            [self.after_cnn_len(l.item()) for l in feature_lens],
            device=feature_lens.device
        )

        total = audio_features.size(0)
        delta = total - post_lens.sum()
        if delta != 0:
            post_lens[-1] += delta

        audio_features = audio_features.split(post_lens.tolist(), dim=0)
        audio_features = pad_sequence(audio_features, batch_first=True) # for Attentive Pooling
        audio_features = audio_features.to(torch.float32)

        mean = audio_features.mean(dim=1)
        std = audio_features.std(dim=1)

        embeddings = torch.cat([mean, std], dim=1)
        return {
            "embeddings": embeddings,
        }

model = Qwen3_Extractor(1)
model.cuda()
model.eval()

parser = train.parse_args()
args = parser.parse_args(["--init", "--model_name", "dev", "--pooling_model",
                          "qwen", "--config_path", "configs/general.json"]) # qwen

model_dir = os.path.join("./logs", args.model_name)
os.makedirs(model_dir, exist_ok=True)
port = None

config_path = args.config_path if args.init else os.path.join(model_dir, "config.json")
hps = train.load_config(config_path, model_dir, args)

hps.data.acoustic_feature = False
hps.data.mean_std_norm = False

df_train, _ = train.load_data(hps)
collate_fn = train.get_collate_fn(hps)
target_labels = df_train[hps.data.target_column]
df_train = clean_by_ast(df_train, threshold=0.95)

_, coda_loader = train.prepare_fold_data(
    df_train, df_train, hps, collate_fn,
    use_precomputed=args.use_precomputed,
    precomputed_dir=args.precomputed_dir
)

df_cirdz = pd.read_csv(f'/run/media/fourier/Data1/Pras/Thesis_Nexus/NXSCough/data/metadata_cirdz.csv.train')
df_cirdz = df_cirdz.reset_index(drop=True)
df_cirdz = df_cirdz[hps.data.column_order]

_, cirdz_loader = train.prepare_fold_data(
    df_cirdz, df_cirdz, hps, collate_fn,
    use_precomputed=args.use_precomputed,
    precomputed_dir=args.precomputed_dir
)

df_tbscreen = pd.read_csv(f'/run/media/fourier/Data1/Pras/DatabaseLLM/TBscreen_Dataset/metadata.csv')
df_tbscreen = df_tbscreen[df_tbscreen['type'] == 'solic']
df_tbscreen["participant"] = df_tbscreen["participant"].astype("category").cat.codes
df_tbscreen = df_tbscreen.reset_index(drop=True)
df_tbscreen = df_tbscreen[hps.data.column_order]
df_tbscreen = clean_by_ast(df_tbscreen, threshold=0.95)

_, tbscreen_loader = train.prepare_fold_data(
    df_tbscreen, df_tbscreen, hps, collate_fn,
    use_precomputed=args.use_precomputed,
    precomputed_dir=args.precomputed_dir
)

df_ukcovid = pd.read_csv(f'/run/media/fourier/Data1/Pras/DatabaseLLM/ukcovid19/metadata.csv.train')
df_ukcovid[["weight_loss", "hemoptysis", "night_sweats", "smoker"]] = 0
df_ukcovid = df_ukcovid.rename(columns={"path_file_audio": "path_file", "participant_identifier": "participant", "covid_test_result": "disease_status"})
df_ukcovid["gender"] = df_ukcovid["gender"].map({"Male": 0, "Female": 1})
df_ukcovid["disease_status"] = df_ukcovid["disease_status"].map({4: 0})
df_ukcovid["participant"] = df_ukcovid["participant"].str.lstrip("P").astype(int)
df_ukcovid = df_ukcovid.dropna().reset_index(drop=True)
df_ukcovid = df_ukcovid.sample(n=4000, random_state=42).reset_index(drop=True)
df_ukcovid = df_ukcovid[hps.data.column_order]

_, ukcovid_loader = train.prepare_fold_data(
    df_ukcovid, df_ukcovid, hps, collate_fn,
    use_precomputed=args.use_precomputed,
    precomputed_dir=args.precomputed_dir
)

# %%
train_embeds = []
train_wavs = [] 
with torch.no_grad():
    for idx, batch in tqdm(enumerate(coda_loader), total=len(coda_loader)):
        wavs_names, audio, _, attention_masks, dse_ids, [patient_ids, _, tabular_ids, _]  = batch
        audio = audio.cuda()
        out_model = model(audio, attention_mask=attention_masks)
        embed = out_model['embeddings']

        dse_ids = torch.argmax(dse_ids, dim=1)
        train_wavs.extend(wavs_names)
        train_embeds.append(embed.cpu())

train_wavs = np.array(train_wavs)
train_embeds = torch.cat(train_embeds, dim=0).numpy()

df_train = df_train.set_index("path_file").loc[train_wavs].reset_index()
df_train["embed"] = list(train_embeds)
df_train["db"] = 0

train_embeds = []
train_wavs = []
with torch.no_grad():
    for idx, batch in tqdm(enumerate(cirdz_loader), total=len(cirdz_loader)):
        wavs_names, audio, _, attention_masks, dse_ids, [patient_ids, _, tabular_ids, _]  = batch
        audio = audio.cuda()
        out_model = model(audio, attention_mask=attention_masks)
        embed = out_model['embeddings']

        dse_ids = torch.argmax(dse_ids, dim=1)
        train_wavs.extend(wavs_names)
        train_embeds.append(embed.cpu())

train_wavs = np.array(train_wavs)
train_embeds = torch.cat(train_embeds, dim=0).numpy()

df_cirdz = df_cirdz.set_index("path_file").loc[train_wavs].reset_index()
df_cirdz["embed"] = list(train_embeds)
df_cirdz["db"] = 1

train_embeds = []
train_wavs = []
with torch.no_grad():
    for idx, batch in tqdm(enumerate(tbscreen_loader), total=len(tbscreen_loader)):
        wavs_names, audio, _, attention_masks, dse_ids, [patient_ids, _, tabular_ids, _]  = batch
        audio = audio.cuda()
        out_model = model(audio, attention_mask=attention_masks)
        embed = out_model['embeddings']

        dse_ids = torch.argmax(dse_ids, dim=1)
        train_wavs.extend(wavs_names)
        train_embeds.append(embed.cpu())

train_wavs = np.array(train_wavs)
train_embeds = torch.cat(train_embeds, dim=0).numpy()

df_tbscreen = df_tbscreen.set_index("path_file").loc[train_wavs].reset_index()
df_tbscreen["embed"] = list(train_embeds)
df_tbscreen["db"] = 2

# train_embeds = []
# train_wavs = []
# with torch.no_grad():
#     for idx, batch in tqdm(enumerate(ukcovid_loader), total=len(ukcovid_loader)):
#         wavs_names, audio, _, attention_masks, dse_ids, [patient_ids, _, tabular_ids, _]  = batch
#         audio = audio.cuda()
#         out_model = model(audio, attention_mask=attention_masks)
#         embed = out_model['embeddings']

#         dse_ids = torch.argmax(dse_ids, dim=1)
#         train_wavs.extend(wavs_names)
#         train_embeds.append(embed.cpu())

# train_wavs = np.array(train_wavs)
# train_embeds = torch.cat(train_embeds, dim=0).numpy()

# df_ukcovid = df_ukcovid.set_index("path_file").loc[train_wavs].reset_index()
# df_ukcovid["embed"] = list(train_embeds)
# df_ukcovid["db"] = 3

df_combine = pd.concat([df_train, df_cirdz, df_tbscreen]).reset_index(drop=True)

# %%
del feature_extractor, model_ast, model, ukcovid_loader, cirdz_loader, coda_loader, audio
gc.collect()
if torch.cuda.is_available():
    torch.cuda.empty_cache()

# %%
# summary, model_results = run_embedding_classification(
#     df_combine,
#     train_db=0,
#     test_dbs=[1, 2],
#     label_col="disease_status",
# )
# summary

# %% [markdown]
# ## Outlier

# %%
def get_outlier_mask(df_temp, contamination=0.05):
    """
    df_temp: already filtered dataframe (single class / subset)
    contamination: fraction of outliers (e.g. 0.05)

    Returns:
        outlier_mask (True = outlier)
    """

    # ─────────────────────────────────────────────
    # Embedding + normalize
    # ─────────────────────────────────────────────
    df_temp = df_temp.reset_index(drop=True)
    raw_emb = np.stack(df_temp["embed"].values)

    # ─────────────────────────────────────────────
    # PCA (with whitening for better distance behavior)
    # ─────────────────────────────────────────────
    pca = PCA(n_components=min(50, raw_emb.shape[1]), random_state=42, whiten=True)
    emb_pca = pca.fit_transform(raw_emb)

    # ─────────────────────────────────────────────
    # Isolation Forest
    # ─────────────────────────────────────────────
    clf = IsolationForest(
        n_estimators=200,
        contamination=contamination,
        random_state=42,
        n_jobs=-1
    )
    clf.fit(emb_pca)

    if_scores = -clf.score_samples(emb_pca)
    if_scores = (if_scores - if_scores.min()) / (if_scores.max() - if_scores.min() + 1e-8)

    # ─────────────────────────────────────────────
    # Centroid distance (robust)
    # ─────────────────────────────────────────────
    centroid = emb_pca.mean(axis=0)
    dists = np.linalg.norm(emb_pca - centroid, axis=1)

    med = np.median(dists)
    mad = np.median(np.abs(dists - med)) + 1e-8

    z = (dists - med) / mad
    z = np.clip(z, 0, None)

    cd_scores = (z - z.min()) / (z.max() - z.min() + 1e-8)
    # ─────────────────────────────────────────────
    # Ensemble 
    # ─────────────────────────────────────────────
    scores = 0.5 * if_scores + 0.5 * cd_scores

    threshold = np.percentile(scores, 100 * (1 - contamination))
    outlier_mask = scores >= threshold

    return outlier_mask

# %%
global_outlier_mask = np.zeros(len(df_combine), dtype=bool)
for db in tqdm([0, 1, 2]):
    for ds in [0, 1]:
        subset = (df_combine["db"] == db) & (df_combine["disease_status"] == ds)
        idx = np.where(subset)[0]
        if len(idx) == 0:
            continue

        mask_local = get_outlier_mask(df_combine[subset], contamination=0.05)
        global_outlier_mask[idx] = mask_local

inlier_mask = ~global_outlier_mask

raw_emb = np.stack(df_combine["embed"].values)
#raw_emb = raw_emb / (np.linalg.norm(raw_emb, axis=1, keepdims=True) + 1e-12)
df_combine = df_combine[~global_outlier_mask].reset_index(drop=True)

# %%
raw_emb = np.stack(df_combine['embed'].values)
#raw_emb = raw_emb / (np.linalg.norm(raw_emb, axis=1, keepdims=True) + 1e-12)
db_labels = df_combine['db'].values
disease_statuss = df_combine['disease_status'].values

N, D = raw_emb.shape
print(f"  Embeddings: {N} samples × {D} dims")
print(f"  Databases : {dict(zip(*np.unique(db_labels, return_counts=True)))}")
print(f"  Disease   : {dict(zip(*np.unique(disease_statuss, return_counts=True)))}")

# %%
covariates = disease_statuss.astype(float).reshape(-1, 1)
combat_emb = combat_harmonize(raw_emb, db_labels, covariates=None)
embeddings = {"raw": raw_emb, "ComBat": combat_emb}
df_combine["embed"] = list(combat_emb)


# %%
counts = df_combine["participant"].value_counts()
valid_speakers = counts[counts >= 3].index
df_combine = df_combine[df_combine["participant"].isin(valid_speakers)].reset_index(drop=True)

df_current = df_combine

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import roc_auc_score
from copy import deepcopy
from tqdm import tqdm
import json
import os
from datetime import datetime


import pickle

def save_checkpoint(path, state):
    tmp_path = path + ".tmp"
    with open(tmp_path, "wb") as f:   # binary mode
        pickle.dump(state, f)
    os.replace(tmp_path, path)  # atomic write


def load_checkpoint(path):
    with open(path, "rb") as f:   # binary mode
        state = pickle.load(f)

    print("\n=== Resuming from checkpoint ===")
    print(f"  Saved at       : {state['timestamp']}")
    print(f"  Iteration      : {state['iteration']}")
    print(f"  Global rank    : {state['global_rank']}")
    print(f"  Baseline AUROC : {state['baseline_auroc']:.4f}")
    print(f"  Best AUROC     : {state['best_auroc']:.4f}")
    print(f"  Removed so far : {len(state['removal_log'])}")
    print(f"  Remaining      : {len(state['remaining'])}")
    print("================================\n")

    return state


def print_summary(removal_log, baseline_auroc, best_auroc, remaining):
    print("\n=== Current Summary ===")
    print(f"  Best AUROC achieved : {best_auroc:.4f}")
    print(f"  Current AUROC       : {baseline_auroc:.4f}")
    print(f"  Removed (Label Only): {len(removal_log)}")
    print(f"  Surviving (True TB) : {len(remaining)}")
    if removal_log:
        df_log = pd.DataFrame(removal_log)
        print("\n  Top 5 removed participants (most Label Only):")
        print(
            df_log.sort_values("removal_order")[
                ["participant", "removal_order", "delta_auroc", "cumulative_delta"]
            ].head(5).to_string(index=False)
        )
    print("=======================\n")


def compute_auroc_cv(df, classifier, n_splits=5):
    """Compute mean AUROC. Accepts single classifier or dict of classifiers."""
    if isinstance(classifier, dict):
        return np.mean([
            compute_auroc_cv(df, clf, n_splits)
            for clf in classifier.values()
        ])

    X = np.stack(df["embed"].values)
    y = df["disease_status"].values
    groups = df["participant"].values

    cv = StratifiedGroupKFold(n_splits=n_splits)
    aucs = []

    for train_idx, val_idx in cv.split(X, y, groups):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        model = deepcopy(classifier)
        model.fit(X_train, y_train)
        proba = model.predict_proba(X_val)[:, 1]
        aucs.append(roc_auc_score(y_val, proba))

    return np.mean(aucs)


def greedy_participant_scoring(
    df_current,
    classifier,
    n_splits=5,
    batch_pct=0.25,
    min_batch=1,
    checkpoint_path="greedy_checkpoint.json",
    summary_every=1,       # print summary every N iterations
):
    # --- Load or initialize state ---
    if checkpoint_path and os.path.exists(checkpoint_path):
        state = load_checkpoint(checkpoint_path)
        removal_log   = state["removal_log"]
        remaining     = state["remaining"]
        baseline_auroc = state["baseline_auroc"]
        cumulative_delta = state["cumulative_delta"]
        iteration     = state["iteration"]
        global_rank   = state["global_rank"]
        best_auroc    = state["best_auroc"]

        # Reconstruct df_work by dropping already-removed participants
        removed_set = {entry["participant"] for entry in removal_log}
        df_work = df_current[~df_current["participant"].isin(removed_set)].copy()

    else:
        df_work = df_current.copy()
        remaining = list(df_work[df_work["disease_status"] == 1]["participant"].unique())
        baseline_auroc = compute_auroc_cv(df_work, classifier, n_splits)
        removal_log = []
        cumulative_delta = 0.0
        iteration = 1
        global_rank = 1
        best_auroc = baseline_auroc
        print(f"Baseline AUROC: {baseline_auroc:.4f} | Participants: {len(remaining)}")

    total_participants = len(remaining) + len(removal_log)
    outer_bar = tqdm(
        total=total_participants,
        initial=len(removal_log),
        desc="Participants Processed",
        unit="participant"
    )

    # --- Main loop ---
    while remaining:
        candidate_results = []

        inner_bar = tqdm(
            remaining,
            desc=f"Iter {iteration} | Scoring",
            unit="participant",
            leave=False
        )
        for p in inner_bar:
            df_candidate = df_work[df_work["participant"] != p]
            auroc = compute_auroc_cv(df_candidate, classifier, n_splits)
            delta = auroc - baseline_auroc
            candidate_results.append((p, auroc, delta))
            inner_bar.set_postfix({"participant": p, "delta": f"{delta:.4f}"})
        inner_bar.close()

        candidate_results.sort(key=lambda x: x[2], reverse=True)
        positive_candidates = [(p, a, d) for p, a, d in candidate_results if d > 0]

        if not positive_candidates:
            tqdm.write(f"Iteration {iteration}: No removal improves AUROC. Stopping.")
            break

        batch_size = max(min_batch, int(np.ceil(len(remaining) * batch_pct)))
        batch_size = min(batch_size, len(positive_candidates))
        batch = positive_candidates[:batch_size]

        tqdm.write(
            f"Iteration {iteration}: Removing {batch_size} participants | "
            f"Delta range=[{batch[-1][2]:.4f}, {batch[0][2]:.4f}] | "
            f"Remaining after={len(remaining) - batch_size}"
        )

        for rank_in_batch, (p, auroc, delta) in enumerate(batch):
            cumulative_delta += delta
            removal_log.append({
                "participant": p,
                "removal_order": global_rank,
                "iteration": iteration,
                "rank_in_batch": rank_in_batch + 1,
                "delta_auroc": round(delta, 6),
                "cumulative_delta": round(cumulative_delta, 6),
                "auroc_after_removal": round(auroc, 6),
                "label": "Label Only"
            })
            df_work = df_work[df_work["participant"] != p]
            remaining.remove(p)
            global_rank += 1
            outer_bar.update(1)

        # Recompute baseline on cleaned df_work
        baseline_auroc = compute_auroc_cv(df_work, classifier, n_splits)
        best_auroc = max(best_auroc, baseline_auroc)

        tqdm.write(f"  → New baseline AUROC: {baseline_auroc:.4f} | Best so far: {best_auroc:.4f}")
        outer_bar.set_postfix({"auroc": f"{baseline_auroc:.4f}", "best": f"{best_auroc:.4f}", "iter": iteration})

        # --- Checkpoint ---
        if checkpoint_path:
            save_checkpoint(checkpoint_path, {
                "iteration": iteration + 1,
                "global_rank": global_rank,
                "baseline_auroc": baseline_auroc,
                "best_auroc": best_auroc,
                "cumulative_delta": cumulative_delta,
                "removal_log": removal_log,
                "remaining": remaining,
                "timestamp": datetime.now().isoformat(),
            })

        # --- Periodic summary ---
        if iteration % summary_every == 0:
            print_summary(removal_log, baseline_auroc, best_auroc, remaining)

        iteration += 1

    outer_bar.update(len(remaining))
    outer_bar.close()

    # Survivors
    true_tb = [
        {
            "participant": p,
            "removal_order": None,
            "iteration": None,
            "rank_in_batch": None,
            "delta_auroc": None,
            "cumulative_delta": 0.0,
            "auroc_after_removal": None,
            "label": "True TB Biomarker"
        }
        for p in remaining
    ]

    print_summary(removal_log, baseline_auroc, best_auroc, remaining)

    df_scores = pd.DataFrame(removal_log + true_tb)
    return df_scores

# First run (or resume automatically if checkpoint exists)
df_scores = greedy_participant_scoring(
    df_current,
    classifier=list(get_classifiers().values())[0],
    n_splits=5,
    batch_pct=0.25,
    min_batch=1,
    checkpoint_path="greedy_checkpoint.json",
    summary_every=2,   # print summary every 2 iterations
)

# Peek at early results anytime without running
import json
with open("greedy_checkpoint.json") as f:
    state = json.load(f)
pd.DataFrame(state["removal_log"])
