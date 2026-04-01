import csv
import os
from dataclasses import dataclass, field
from typing import Dict, List, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.model_selection import StratifiedKFold
from sksurv.ensemble import RandomSurvivalForest
from sksurv.metrics import concordance_index_censored
from sksurv.util import Surv


CLINICAL_COLUMNS = [
    'Age', 'Gender', 'Tobacco Consumption', 'Alcohol Consumption', 'Performance Status',
    'Treatment', 'T-stage', 'N-stage', 'M-stage', 'HPV Status', 'CenterID'
]
RESERVED_COLUMNS = {'PatientID', 'RFS', 'Relapse'}
TUMOR_IMAGE_PREFIXES = ('tumor_', 'pet_original_', 'ct_original_')
ATLAS_PREFIX = 'atlas__'


@dataclass
class FeatureBlockSpec:
    name: str
    columns: List[str]
    max_features: int


@dataclass
class ModelSpec:
    name: str
    title: str
    locked_columns: List[str]
    feature_blocks: List[FeatureBlockSpec] = field(default_factory=list)


@dataclass
class ModelResult:
    name: str
    title: str
    n_samples: int
    n_events: int
    c_index_oof: float
    fold_c_index_mean: float
    fold_c_index_std: float
    clinical_count: int
    imaging_count: int
    selected_feature_count_mean: float
    output_dir: str


def load_case_feature_table(feature_csv: str) -> pd.DataFrame:
    if not os.path.exists(feature_csv):
        raise FileNotFoundError(
            f'Feature table not found: {feature_csv}. Run the corresponding feature-extraction script first.'
        )
    df = pd.read_csv(feature_csv)
    required = {'PatientID', 'RFS', 'Relapse'}
    if not required.issubset(df.columns):
        raise RuntimeError(f'Feature CSV must contain columns: {sorted(required)}')
    df['PatientID'] = df['PatientID'].astype(str)
    return df


def get_clinical_columns(df: pd.DataFrame) -> List[str]:
    return [col for col in CLINICAL_COLUMNS if col in df.columns]


def get_imaging_columns(df: pd.DataFrame) -> List[str]:
    clinical_cols = set(get_clinical_columns(df))
    return [col for col in df.columns if col not in RESERVED_COLUMNS and col not in clinical_cols]


def get_tumor_columns(df: pd.DataFrame) -> List[str]:
    return [
        col for col in get_imaging_columns(df)
        if col.startswith(TUMOR_IMAGE_PREFIXES)
    ]


def get_atlas_only_columns(df: pd.DataFrame) -> List[str]:
    return [
        col for col in get_imaging_columns(df)
        if not col.startswith(TUMOR_IMAGE_PREFIXES)
    ]


def get_atlas_pet_columns(df: pd.DataFrame) -> List[str]:
    return [col for col in df.columns if col.startswith(ATLAS_PREFIX) and '_pet_' in col]


def topk_mean_rows(values: np.ndarray, k: int) -> np.ndarray:
    if values.ndim != 2 or values.shape[1] == 0:
        return np.full(values.shape[0], np.nan, dtype=float)
    out = np.full(values.shape[0], np.nan, dtype=float)
    k = max(1, int(k))
    for row_index in range(values.shape[0]):
        row = values[row_index]
        row = row[np.isfinite(row)]
        if row.size == 0:
            continue
        kk = min(k, row.size)
        order = np.argsort(row)
        out[row_index] = float(np.mean(row[order[-kk:]]))
    return out


def add_atlas_pet_summary_features(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    pet_cols = get_atlas_pet_columns(df)
    suffixes = ('pet_mean', 'pet_p90', 'pet_p95', 'pet_max')
    summary_columns: Dict[str, np.ndarray] = {}
    for suffix in suffixes:
        cols = [col for col in pet_cols if col.endswith(suffix)]
        if not cols:
            continue
        values = df[cols].apply(pd.to_numeric, errors='coerce').to_numpy(dtype=float)
        summary_columns[f'atlas_summary_{suffix}_mean'] = np.nanmean(values, axis=1)
        summary_columns[f'atlas_summary_{suffix}_std'] = np.nanstd(values, axis=1)
        summary_columns[f'atlas_summary_{suffix}_max'] = np.nanmax(values, axis=1)
        summary_columns[f'atlas_summary_{suffix}_p90'] = np.nanpercentile(values, 90.0, axis=1)
        summary_columns[f'atlas_summary_{suffix}_top3_mean'] = topk_mean_rows(values, 3)
        summary_columns[f'atlas_summary_{suffix}_top5_mean'] = topk_mean_rows(values, 5)
        summary_columns[f'atlas_summary_{suffix}_gt1_count'] = np.sum(values > 1.0, axis=1).astype(float)
        summary_columns[f'atlas_summary_{suffix}_gt1p5_count'] = np.sum(values > 1.5, axis=1).astype(float)
        summary_columns[f'atlas_summary_{suffix}_gt2_count'] = np.sum(values > 2.0, axis=1).astype(float)
        summary_columns[f'atlas_summary_{suffix}_excess_gt1_sum'] = np.nansum(np.clip(values - 1.0, a_min=0.0, a_max=None), axis=1)

    if not summary_columns:
        return df.copy(), []

    summary_df = pd.DataFrame(summary_columns, index=df.index)
    out_df = pd.concat([df.copy(), summary_df], axis=1)
    return out_df, list(summary_df.columns)


def merge_tumor_and_atlas_tables(tumor_csv: str, atlas_csv: str) -> pd.DataFrame:
    tumor_df = load_case_feature_table(tumor_csv)
    atlas_df = load_case_feature_table(atlas_csv)

    tumor_clinical = get_clinical_columns(tumor_df)
    atlas_clinical = get_clinical_columns(atlas_df)
    clinical_cols = [col for col in CLINICAL_COLUMNS if col in tumor_clinical or col in atlas_clinical]

    tumor_cols = get_tumor_columns(tumor_df)
    atlas_only_cols = get_atlas_only_columns(atlas_df)
    if not atlas_only_cols:
        raise RuntimeError(
            'No atlas-only imaging columns were found. Rerun the atlas extraction script so the case table contains atlas CTSeg biomarkers '
            'and does not rely only on tumor-derived features.'
        )

    tumor_keep = ['PatientID', 'RFS', 'Relapse'] + [col for col in clinical_cols if col in tumor_df.columns] + tumor_cols
    atlas_keep = ['PatientID', 'RFS', 'Relapse'] + [col for col in clinical_cols if col in atlas_df.columns] + atlas_only_cols
    tumor_use = tumor_df[tumor_keep].copy()
    atlas_use = atlas_df[atlas_keep].copy()
    atlas_use = atlas_use.rename(columns={col: f'{ATLAS_PREFIX}{col}' for col in atlas_only_cols})

    merged = pd.merge(tumor_use, atlas_use, on='PatientID', suffixes=('_tumor', '_atlas'), how='inner')
    if merged.empty:
        raise RuntimeError('No overlapping PatientID values were found between the tumor and atlas feature tables.')

    rfs_t = pd.to_numeric(merged['RFS_tumor'], errors='coerce').to_numpy(dtype=float)
    rfs_a = pd.to_numeric(merged['RFS_atlas'], errors='coerce').to_numpy(dtype=float)
    rel_t = pd.to_numeric(merged['Relapse_tumor'], errors='coerce').to_numpy(dtype=float)
    rel_a = pd.to_numeric(merged['Relapse_atlas'], errors='coerce').to_numpy(dtype=float)
    outcome_ok = (
        np.isfinite(rfs_t) & np.isfinite(rfs_a) & np.isclose(rfs_t, rfs_a, atol=1e-6) &
        np.isfinite(rel_t) & np.isfinite(rel_a) & (rel_t.astype(int) == rel_a.astype(int))
    )
    if not np.all(outcome_ok):
        bad_ids = merged.loc[~outcome_ok, 'PatientID'].astype(str).tolist()[:10]
        raise RuntimeError(f'Tumor and atlas tables disagree on RFS/Relapse for some patients, e.g. {bad_ids}')

    output_columns: Dict[str, pd.Series] = {
        'PatientID': merged['PatientID'].astype(str),
        'RFS': pd.Series(rfs_t, index=merged.index, dtype=float),
        'Relapse': pd.Series(rel_t.astype(int), index=merged.index, dtype=int),
    }

    for col in clinical_cols:
        tumor_name = f'{col}_tumor' if f'{col}_tumor' in merged.columns else None
        atlas_name = f'{col}_atlas' if f'{col}_atlas' in merged.columns else None
        series = None
        if tumor_name is not None:
            series = pd.to_numeric(merged[tumor_name], errors='coerce')
        if atlas_name is not None:
            atlas_series = pd.to_numeric(merged[atlas_name], errors='coerce')
            if series is None:
                series = atlas_series
            else:
                series = series.where(series.notna(), atlas_series)
        if series is not None:
            output_columns[col] = series

    for col in tumor_cols:
        output_columns[col] = pd.to_numeric(merged[col], errors='coerce')
    atlas_prefixed_cols = [f'{ATLAS_PREFIX}{col}' for col in atlas_only_cols]
    for col in atlas_prefixed_cols:
        output_columns[col] = pd.to_numeric(merged[col], errors='coerce')
    return pd.DataFrame(output_columns).copy()


def concordance_index_value(event: np.ndarray, time: np.ndarray, risk: np.ndarray) -> float:
    valid = np.isfinite(time) & np.isfinite(risk)
    if valid.sum() == 0:
        return np.nan
    event_bool = np.asarray(event, dtype=bool)[valid]
    time_val = np.asarray(time, dtype=float)[valid]
    risk_val = np.asarray(risk, dtype=float)[valid]
    if len(np.unique(event_bool.astype(int))) < 2 and event_bool.sum() == 0:
        return np.nan
    return float(concordance_index_censored(event_bool, time_val, risk_val)[0])


def plot_km_with_ticks_two_groups(
    time_days: np.ndarray,
    event: np.ndarray,
    grp_high: np.ndarray,
    out_path: str,
    title: str,
    censor_merge_window_days: float = 45.0,
) -> None:
    def merge_censor_ticks(times: np.ndarray, levels: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        if times.size == 0 or censor_merge_window_days <= 0:
            return times, levels
        order = np.argsort(times)
        times = times[order]
        levels = levels[order]
        merged_times: List[float] = []
        merged_levels: List[float] = []
        cluster_times = [float(times[0])]
        cluster_levels = [float(levels[0])]
        for idx in range(1, times.size):
            if float(times[idx]) - cluster_times[-1] <= float(censor_merge_window_days):
                cluster_times.append(float(times[idx]))
                cluster_levels.append(float(levels[idx]))
            else:
                merged_times.append(float(np.mean(cluster_times)))
                merged_levels.append(float(np.mean(cluster_levels)))
                cluster_times = [float(times[idx])]
                cluster_levels = [float(levels[idx])]
        merged_times.append(float(np.mean(cluster_times)))
        merged_levels.append(float(np.mean(cluster_levels)))
        return np.asarray(merged_times, dtype=float), np.asarray(merged_levels, dtype=float)

    def km_curve(t: np.ndarray, e: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        order = np.argsort(t)
        t = np.asarray(t, dtype=float)[order]
        e = np.asarray(e, dtype=int)[order]
        uniq_times = np.unique(t)
        n_at_risk = float(len(t))
        surv = 1.0
        xs = [0.0]
        ys = [1.0]
        censor_times: List[float] = []
        censor_levels: List[float] = []
        for ut in uniq_times:
            mask = (t == ut)
            d = float((e[mask] == 1).sum())
            c = float((e[mask] == 0).sum())
            if n_at_risk > 0 and d > 0:
                surv *= (1.0 - d / n_at_risk)
                xs.extend([float(ut), float(ut)])
                ys.extend([ys[-1], surv])
            if c > 0:
                censor_times.extend([float(ut)] * int(c))
                censor_levels.extend([surv] * int(c))
            n_at_risk -= (d + c)
        ct, cl = merge_censor_ticks(np.asarray(censor_times, dtype=float), np.asarray(censor_levels, dtype=float))
        return np.asarray(xs, dtype=float), np.asarray(ys, dtype=float), ct, cl

    mask_fin = np.isfinite(time_days)
    t = np.asarray(time_days, dtype=float)[mask_fin]
    e = np.asarray(event, dtype=int)[mask_fin]
    high = np.asarray(grp_high, dtype=bool)[mask_fin]
    if high.sum() == 0 or (~high).sum() == 0:
        return
    xh, yh, cth, cyh = km_curve(t[high], e[high])
    xl, yl, ctl, cyl = km_curve(t[~high], e[~high])
    plt.figure(figsize=(6.5, 4.5), dpi=140)
    plt.step(xh, yh, where='post', color='tab:red', label='High-risk group')
    plt.step(xl, yl, where='post', color='tab:blue', label='Low-risk group')
    if cth.size:
        plt.scatter(cth, cyh, marker='|', color='tab:red', s=80, linewidths=1.5)
    if ctl.size:
        plt.scatter(ctl, cyl, marker='|', color='tab:blue', s=80, linewidths=1.5)
    plt.ylim(0.0, 1.05)
    plt.xlabel('Time (days)')
    plt.ylabel('Relapse-free survival probability')
    plt.title(title)
    plt.grid(alpha=0.25)
    plt.legend(loc='best')
    try:
        plt.savefig(out_path, bbox_inches='tight')
    finally:
        plt.close()


def plot_cindex_comparison(results: Sequence[ModelResult], out_path: str, title: str) -> None:
    if not results:
        return
    labels = [result.name.replace('_', '\n') for result in results]
    scores = [result.c_index_oof for result in results]
    errors = [result.fold_c_index_std for result in results]
    plt.figure(figsize=(max(6.5, 1.8 * len(results)), 4.5), dpi=140)
    plt.bar(np.arange(len(results)), scores, yerr=errors, color='tab:green', alpha=0.85, capsize=4)
    plt.axhline(0.5, linestyle='--', color='grey', linewidth=1)
    plt.xticks(np.arange(len(results)), labels)
    plt.ylabel('Out-of-fold C-index')
    plt.title(title)
    plt.ylim(0.0, min(1.0, max(scores + [0.6]) + 0.08))
    plt.grid(axis='y', alpha=0.25)
    try:
        plt.savefig(out_path, bbox_inches='tight')
    finally:
        plt.close()


def greedy_correlation_prune(X: np.ndarray, feature_names: Sequence[str], threshold: float) -> Tuple[np.ndarray, List[str]]:
    if X.ndim != 2 or X.shape[1] <= 1 or threshold is None or threshold >= 1.0:
        return np.arange(X.shape[1], dtype=int), list(feature_names)
    corr = np.corrcoef(X, rowvar=False)
    if np.ndim(corr) != 2:
        return np.arange(X.shape[1], dtype=int), list(feature_names)
    corr = np.nan_to_num(corr, nan=0.0, posinf=0.0, neginf=0.0)
    keep: List[int] = []
    dropped = np.zeros(X.shape[1], dtype=bool)
    for idx in range(X.shape[1]):
        if dropped[idx]:
            continue
        keep.append(idx)
        correlated = np.abs(corr[idx]) >= float(threshold)
        correlated[idx] = False
        dropped |= correlated
    keep_array = np.asarray(keep, dtype=int)
    return keep_array, [feature_names[idx] for idx in keep_array]


def rank_features_by_univariate_cindex(X_train: np.ndarray, y_train) -> np.ndarray:
    scores = np.zeros(X_train.shape[1], dtype=float)
    for index in range(X_train.shape[1]):
        values = np.asarray(X_train[:, index], dtype=float)
        if (not np.all(np.isfinite(values))) or np.nanstd(values) <= 1e-9:
            scores[index] = 0.0
            continue
        try:
            c_value = float(concordance_index_censored(y_train['event'], y_train['time'], values)[0])
        except Exception:
            c_value = np.nan
        if not np.isfinite(c_value):
            scores[index] = 0.0
        else:
            scores[index] = abs(c_value - 0.5)
    return scores


def unique_preserve_order(columns: Sequence[str]) -> List[str]:
    seen = set()
    ordered: List[str] = []
    for column in columns:
        if column in seen:
            continue
        seen.add(column)
        ordered.append(column)
    return ordered


def select_feature_block(
    X_train: np.ndarray,
    X_test: np.ndarray,
    feature_names: Sequence[str],
    y_train,
    correlation_threshold: float,
    max_features: int,
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    if X_train.shape[1] == 0:
        return X_train, X_test, []

    keep_corr, kept_names = greedy_correlation_prune(X_train, feature_names, correlation_threshold)
    X_train_sel = X_train[:, keep_corr]
    X_test_sel = X_test[:, keep_corr]

    if int(max_features) > 0 and X_train_sel.shape[1] > int(max_features):
        scores = rank_features_by_univariate_cindex(X_train_sel, y_train)
        order = np.argsort(scores)[::-1][:int(max_features)]
        X_train_sel = X_train_sel[:, order]
        X_test_sel = X_test_sel[:, order]
        kept_names = [kept_names[idx] for idx in order]

    return X_train_sel, X_test_sel, kept_names


def cross_validated_rsf(
    df: pd.DataFrame,
    model_spec: ModelSpec,
    output_dir: str,
    n_splits: int,
    n_estimators: int,
    min_samples_leaf: int,
    correlation_threshold: float,
    random_state: int,
) -> ModelResult:
    os.makedirs(output_dir, exist_ok=True)

    time_np = pd.to_numeric(df['RFS'], errors='coerce').to_numpy(dtype=float)
    event_np = pd.to_numeric(df['Relapse'], errors='coerce').fillna(-1).to_numpy(dtype=int)
    valid_rows = np.isfinite(time_np) & np.isin(event_np, [0, 1])
    if valid_rows.sum() < n_splits:
        raise RuntimeError(f'Not enough valid samples ({int(valid_rows.sum())}) for {n_splits}-fold CV in model {model_spec.name}.')

    model_columns = unique_preserve_order(
        list(model_spec.locked_columns) + [col for block in model_spec.feature_blocks for col in block.columns]
    )
    work_df = df.loc[valid_rows, ['PatientID', 'RFS', 'Relapse'] + model_columns].reset_index(drop=True)
    patient_ids = work_df['PatientID'].astype(str).tolist()
    time_used = pd.to_numeric(work_df['RFS'], errors='coerce').to_numpy(dtype=float)
    event_used = pd.to_numeric(work_df['Relapse'], errors='coerce').to_numpy(dtype=int)
    X_df = work_df[model_columns].apply(pd.to_numeric, errors='coerce')
    y = Surv.from_arrays(event=event_used.astype(bool), time=time_used.astype(float))

    splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    oof_risk = np.full(X_df.shape[0], np.nan, dtype=float)
    fold_rows: List[List[object]] = []
    feature_frequency: Dict[str, int] = {}

    for fold_idx, (train_idx, test_idx) in enumerate(splitter.split(X_df, event_used), start=1):
        X_train_df = X_df.iloc[train_idx].copy()
        X_test_df = X_df.iloc[test_idx].copy()
        y_train = y[train_idx]
        y_test = y[test_idx]

        nonempty = ~X_train_df.isna().all(axis=0).to_numpy()
        if nonempty.sum() == 0:
            raise RuntimeError(f'All features were empty in training fold {fold_idx} for model {model_spec.name}.')
        X_train_df = X_train_df.iloc[:, nonempty]
        X_test_df = X_test_df.iloc[:, nonempty]

        imputer = SimpleImputer(strategy='median')
        X_train = imputer.fit_transform(X_train_df)
        X_test = imputer.transform(X_test_df)
        feature_names = list(X_train_df.columns)

        stds = np.nanstd(X_train, axis=0)
        keep_var = np.isfinite(stds) & (stds > 1e-9)
        if keep_var.sum() == 0:
            raise RuntimeError(f'All features were removed by the variance filter in fold {fold_idx} for model {model_spec.name}.')
        X_train = X_train[:, keep_var]
        X_test = X_test[:, keep_var]
        feature_names = [feature_names[idx] for idx in np.where(keep_var)[0]]

        name_to_index = {name: idx for idx, name in enumerate(feature_names)}
        locked_keep_names = [name for name in model_spec.locked_columns if name in name_to_index]
        locked_indices = [name_to_index[name] for name in locked_keep_names]

        selected_train_parts: List[np.ndarray] = []
        selected_test_parts: List[np.ndarray] = []
        selected_feature_names: List[str] = []

        if locked_indices:
            selected_train_parts.append(X_train[:, locked_indices])
            selected_test_parts.append(X_test[:, locked_indices])
            selected_feature_names.extend(locked_keep_names)

        for block in model_spec.feature_blocks:
            block_names = [name for name in block.columns if name in name_to_index]
            if not block_names:
                continue
            block_indices = [name_to_index[name] for name in block_names]
            X_train_block = X_train[:, block_indices]
            X_test_block = X_test[:, block_indices]
            X_train_block, X_test_block, block_selected_names = select_feature_block(
                X_train_block,
                X_test_block,
                block_names,
                y_train,
                correlation_threshold,
                block.max_features,
            )
            if block_selected_names:
                selected_train_parts.append(X_train_block)
                selected_test_parts.append(X_test_block)
                selected_feature_names.extend(block_selected_names)

        if not selected_train_parts:
            raise RuntimeError(f'No usable features remained after block selection in fold {fold_idx} for model {model_spec.name}.')

        X_train = np.concatenate(selected_train_parts, axis=1)
        X_test = np.concatenate(selected_test_parts, axis=1)

        model = RandomSurvivalForest(
            n_estimators=int(n_estimators),
            min_samples_leaf=int(min_samples_leaf),
            max_features='sqrt',
            n_jobs=-1,
            random_state=int(random_state) + fold_idx,
        )
        model.fit(X_train, y_train)
        risk = np.asarray(model.predict(X_test), dtype=float)
        oof_risk[test_idx] = risk

        fold_c = concordance_index_value(y_test['event'], y_test['time'], risk)
        fold_rows.append([
            fold_idx,
            len(train_idx),
            len(test_idx),
            int(np.sum(y_test['event'])),
            float(fold_c),
            len(selected_feature_names),
        ])
        for feature_name in selected_feature_names:
            feature_frequency[feature_name] = feature_frequency.get(feature_name, 0) + 1
        print(
            f'Fold {fold_idx}/{n_splits}: model={model_spec.name} | test_n={len(test_idx)} | '
            f'events={int(np.sum(y_test["event"]))} | C-index={fold_c:.3f} | features={len(selected_feature_names)}'
        )

    c_index = concordance_index_value(event_used, time_used, oof_risk)
    fold_c_values = np.asarray([float(row[4]) for row in fold_rows], dtype=float)
    fold_feature_counts = np.asarray([int(row[5]) for row in fold_rows], dtype=float)

    oof_csv = os.path.join(output_dir, 'rsf_oof_predictions.csv')
    with open(oof_csv, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['PatientID', 'RFS', 'Relapse', 'risk_score'])
        for patient_id, time_val, event_val, risk_val in zip(patient_ids, time_used, event_used, oof_risk):
            writer.writerow([patient_id, float(time_val), int(event_val), float(risk_val)])

    folds_csv = os.path.join(output_dir, 'rsf_cv_folds.csv')
    with open(folds_csv, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['fold', 'train_n', 'test_n', 'test_events', 'c_index', 'selected_features'])
        writer.writerows(fold_rows)

    selected_csv = os.path.join(output_dir, 'rsf_selected_feature_frequency.csv')
    selected_rows = sorted(feature_frequency.items(), key=lambda item: (-item[1], item[0]))
    with open(selected_csv, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['feature', 'selected_in_folds'])
        for feature_name, freq in selected_rows:
            writer.writerow([feature_name, int(freq)])

    km_png = os.path.join(output_dir, 'rsf_km_median.png')
    plot_km_with_ticks_two_groups(
        time_used,
        event_used,
        oof_risk >= np.nanmedian(oof_risk),
        km_png,
        f'{model_spec.title}: KM by RSF risk (median split)',
    )

    summary_path = os.path.join(output_dir, 'rsf_summary.txt')
    with open(summary_path, 'w') as f:
        f.write(f'Model={model_spec.name}\n')
        f.write(f'Title={model_spec.title}\n')
        f.write(f'Samples={X_df.shape[0]}\n')
        f.write(f'Events={int(event_used.sum())}\n')
        f.write(f'Clinical columns locked={len(model_spec.locked_columns)}\n')
        f.write(f'Imaging candidate columns={sum(len(block.columns) for block in model_spec.feature_blocks)}\n')
        f.write(f'Correlation threshold={correlation_threshold:.3f}\n')
        for block in model_spec.feature_blocks:
            f.write(f'Block {block.name}: candidates={len(block.columns)} | max_features={int(block.max_features)}\n')
        f.write(f'RSF trees={int(n_estimators)}\n')
        f.write(f'Min samples leaf={int(min_samples_leaf)}\n')
        f.write(f'OOF C-index={c_index:.3f}\n')
        f.write(f'Mean fold C-index={np.nanmean(fold_c_values):.3f}\n')
        f.write(f'STD fold C-index={np.nanstd(fold_c_values):.3f}\n')
        f.write(f'Mean selected features={np.nanmean(fold_feature_counts):.1f}\n')
        if selected_rows:
            f.write('Top selected features=' + ', '.join(name for name, _ in selected_rows[:20]) + '\n')

    print(f'Model {model_spec.name}: OOF C-index={c_index:.3f}')
    return ModelResult(
        name=model_spec.name,
        title=model_spec.title,
        n_samples=int(X_df.shape[0]),
        n_events=int(event_used.sum()),
        c_index_oof=float(c_index),
        fold_c_index_mean=float(np.nanmean(fold_c_values)),
        fold_c_index_std=float(np.nanstd(fold_c_values)),
        clinical_count=int(len(model_spec.locked_columns)),
        imaging_count=int(sum(len(block.columns) for block in model_spec.feature_blocks)),
        selected_feature_count_mean=float(np.nanmean(fold_feature_counts)),
        output_dir=output_dir,
    )


def run_model_suite(
    df: pd.DataFrame,
    model_specs: Sequence[ModelSpec],
    output_dir: str,
    n_splits: int,
    n_estimators: int,
    min_samples_leaf: int,
    correlation_threshold: float,
    random_state: int,
    suite_title: str,
) -> List[ModelResult]:
    os.makedirs(output_dir, exist_ok=True)
    results: List[ModelResult] = []
    for model_spec in model_specs:
        model_output_dir = os.path.join(output_dir, model_spec.name)
        result = cross_validated_rsf(
            df=df,
            model_spec=model_spec,
            output_dir=model_output_dir,
            n_splits=n_splits,
            n_estimators=n_estimators,
            min_samples_leaf=min_samples_leaf,
            correlation_threshold=correlation_threshold,
            random_state=random_state,
        )
        results.append(result)

    comparison_csv = os.path.join(output_dir, 'rsf_model_comparison.csv')
    with open(comparison_csv, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            'model', 'title', 'samples', 'events', 'oof_c_index', 'mean_fold_c_index', 'std_fold_c_index',
            'clinical_columns', 'imaging_columns', 'mean_selected_features', 'output_dir'
        ])
        for result in results:
            writer.writerow([
                result.name,
                result.title,
                result.n_samples,
                result.n_events,
                result.c_index_oof,
                result.fold_c_index_mean,
                result.fold_c_index_std,
                result.clinical_count,
                result.imaging_count,
                result.selected_feature_count_mean,
                result.output_dir,
            ])

    summary_txt = os.path.join(output_dir, 'rsf_model_comparison.txt')
    ranked = sorted(results, key=lambda item: item.c_index_oof, reverse=True)
    with open(summary_txt, 'w') as f:
        f.write(f'Suite={suite_title}\n')
        f.write(f'Model count={len(results)}\n')
        for result in ranked:
            f.write(
                f'{result.name}: OOF C-index={result.c_index_oof:.3f} | '
                f'mean fold={result.fold_c_index_mean:.3f}±{result.fold_c_index_std:.3f} | '
                f'clinical={result.clinical_count} | imaging={result.imaging_count} | '
                f'mean selected features={result.selected_feature_count_mean:.1f}\n'
            )

    comparison_png = os.path.join(output_dir, 'rsf_model_comparison.png')
    plot_cindex_comparison(results, comparison_png, suite_title)

    print(f'Saved model comparison: {comparison_csv}')
    best = ranked[0] if ranked else None
    if best is not None:
        print(f'Best model: {best.name} | OOF C-index={best.c_index_oof:.3f}')
    return results
