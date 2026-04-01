import argparse
import csv
import os
from dataclasses import dataclass
from typing import Dict, List, Sequence

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.model_selection import StratifiedKFold
from sksurv.ensemble import GradientBoostingSurvivalAnalysis, RandomSurvivalForest
from sksurv.linear_model import CoxPHSurvivalAnalysis
from sksurv.svm import FastSurvivalSVM
from sksurv.util import Surv

from rsf_survival_common import (
    ATLAS_PREFIX,
    add_atlas_pet_summary_features,
    concordance_index_value,
    get_clinical_columns,
    merge_tumor_and_atlas_tables,
    select_feature_block,
    unique_preserve_order,
)


THIS_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_TUMOR_CSV = os.path.join(THIS_DIR, 'population_stats_tumor_radiomics', 'tumor_radiomics_case_features.csv')
DEFAULT_ATLAS_CSV = os.path.join(THIS_DIR, 'population_stats_157lbls', 'imaging_case_features.csv')
DEFAULT_OUTPUT_DIR = os.path.join(THIS_DIR, 'population_stats_survival_model_benchmark_157lbls')
SUMMARY_FAMILY_CHOICES = ('burden', 'overlap', 'z')
MODEL_CHOICES = ('coxph', 'rsf', 'gbsa', 'svm')


@dataclass
class FeatureBlock:
    name: str
    columns: List[str]
    max_features: int


@dataclass
class FeatureSetSpec:
    name: str
    title: str
    locked_columns: List[str]
    feature_blocks: List[FeatureBlock]


@dataclass
class BenchmarkResult:
    model_family: str
    feature_set: str
    title: str
    samples: int
    events: int
    oof_c_index: float
    mean_fold_c_index: float
    std_fold_c_index: float
    mean_selected_features: float
    output_dir: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Benchmark multiple survival models under a matched out-of-fold protocol.')
    parser.add_argument('--tumor-csv', default=DEFAULT_TUMOR_CSV)
    parser.add_argument('--atlas-csv', default=DEFAULT_ATLAS_CSV)
    parser.add_argument('--output-dir', default=DEFAULT_OUTPUT_DIR)
    parser.add_argument('--models', default='coxph,rsf,gbsa,svm')
    parser.add_argument('--n-splits', type=int, default=10)
    parser.add_argument('--correlation-threshold', type=float, default=0.90)
    parser.add_argument('--max-tumor-features', type=int, default=100)
    parser.add_argument('--max-atlas-features', type=int, default=4)
    parser.add_argument('--atlas-feature-mode', choices=['summary', 'raw'], default='summary')
    parser.add_argument('--atlas-summary-families', default='overlap')
    parser.add_argument('--random-state', type=int, default=42)
    parser.add_argument('--cox-alpha', type=float, default=1e-3)
    parser.add_argument('--rsf-n-estimators', type=int, default=500)
    parser.add_argument('--rsf-min-samples-leaf', type=int, default=5)
    parser.add_argument('--gb-learning-rate', type=float, default=0.05)
    parser.add_argument('--gb-n-estimators', type=int, default=300)
    parser.add_argument('--gb-max-depth', type=int, default=2)
    parser.add_argument('--svm-alpha', type=float, default=1.0)
    parser.add_argument('--svm-max-iter', type=int, default=200)
    return parser.parse_args()


def parse_csv_items(raw_value: str, allowed: Sequence[str]) -> List[str]:
    items = [item.strip().lower() for item in str(raw_value).split(',') if item.strip()]
    if not items:
        raise ValueError(f'At least one item expected from {allowed}.')
    bad = [item for item in items if item not in allowed]
    if bad:
        raise ValueError(f'Unsupported values: {bad}. Expected subset of {tuple(allowed)}.')
    seen = set()
    ordered = []
    for item in items:
        if item not in seen:
            ordered.append(item)
            seen.add(item)
    return ordered


def collect_model_columns(df: pd.DataFrame, atlas_feature_mode: str, summary_families: Sequence[str]):
    clinical_cols = get_clinical_columns(df)
    tumor_cols = [col for col in df.columns if col.startswith(('tumor_', 'pet_original_', 'ct_original_'))]

    def collect_prefixed_family(prefix: str):
        merged_prefix = f'{ATLAS_PREFIX}{prefix}'
        return [col for col in df.columns if col.startswith((merged_prefix, prefix))]

    atlas_burden_cols = collect_prefixed_family('atlas_burden_')
    atlas_overlap_cols = collect_prefixed_family('atlas_overlap_')
    atlas_z_cols = collect_prefixed_family('atlas_z_')
    if atlas_feature_mode == 'summary':
        family_to_columns = {
            'burden': list(atlas_burden_cols),
            'overlap': list(atlas_overlap_cols),
            'z': list(atlas_z_cols),
        }
        atlas_cols = []
        for family_name in summary_families:
            atlas_cols.extend(family_to_columns[family_name])
        if not atlas_cols:
            df, atlas_summary_cols = add_atlas_pet_summary_features(df)
            atlas_cols = atlas_summary_cols
    else:
        atlas_cols = [
            col for col in df.columns
            if col.startswith(ATLAS_PREFIX) or col.startswith('atlas_overlap_') or col.startswith('atlas_burden_') or col.startswith('atlas_z_')
        ]
    return df, clinical_cols, tumor_cols, atlas_cols


def build_feature_sets(clinical_cols: Sequence[str], tumor_cols: Sequence[str], atlas_cols: Sequence[str], args: argparse.Namespace):
    return [
        FeatureSetSpec(
            name='clinical_only',
            title='Clinical only',
            locked_columns=list(clinical_cols),
            feature_blocks=[],
        ),
        FeatureSetSpec(
            name='clinical_plus_tumor',
            title='Clinical + tumor imaging',
            locked_columns=list(clinical_cols),
            feature_blocks=[FeatureBlock('tumor', list(tumor_cols), int(args.max_tumor_features))],
        ),
        FeatureSetSpec(
            name='clinical_plus_atlas',
            title='Clinical + atlas biomarkers',
            locked_columns=list(clinical_cols),
            feature_blocks=[FeatureBlock('atlas', list(atlas_cols), int(args.max_atlas_features))],
        ),
        FeatureSetSpec(
            name='clinical_plus_tumor_plus_atlas',
            title='Clinical + tumor imaging + atlas biomarkers',
            locked_columns=list(clinical_cols),
            feature_blocks=[
                FeatureBlock('tumor', list(tumor_cols), int(args.max_tumor_features)),
                FeatureBlock('atlas', list(atlas_cols), int(args.max_atlas_features)),
            ],
        ),
    ]


def instantiate_model(model_family: str, args: argparse.Namespace, fold_idx: int):
    if model_family == 'coxph':
        return CoxPHSurvivalAnalysis(alpha=float(args.cox_alpha), ties='breslow', n_iter=200)
    if model_family == 'rsf':
        return RandomSurvivalForest(
            n_estimators=int(args.rsf_n_estimators),
            min_samples_leaf=int(args.rsf_min_samples_leaf),
            max_features='sqrt',
            n_jobs=-1,
            random_state=int(args.random_state) + int(fold_idx),
        )
    if model_family == 'gbsa':
        return GradientBoostingSurvivalAnalysis(
            loss='coxph',
            learning_rate=float(args.gb_learning_rate),
            n_estimators=int(args.gb_n_estimators),
            max_depth=int(args.gb_max_depth),
            random_state=int(args.random_state) + int(fold_idx),
        )
    if model_family == 'svm':
        return FastSurvivalSVM(
            alpha=float(args.svm_alpha),
            rank_ratio=1.0,
            max_iter=int(args.svm_max_iter),
            random_state=int(args.random_state) + int(fold_idx),
        )
    raise ValueError(f'Unsupported model family: {model_family}')


def maybe_standardize(model_family: str, x_train: np.ndarray, x_test: np.ndarray):
    if model_family not in {'coxph', 'svm'}:
        return x_train, x_test
    mean = np.nanmean(x_train, axis=0)
    std = np.nanstd(x_train, axis=0)
    std = np.where(np.isfinite(std) & (std > 1e-9), std, 1.0)
    return (x_train - mean) / std, (x_test - mean) / std


def cross_validated_benchmark(
    df: pd.DataFrame,
    feature_set: FeatureSetSpec,
    model_family: str,
    output_dir: str,
    args: argparse.Namespace,
) -> BenchmarkResult:
    os.makedirs(output_dir, exist_ok=True)

    time_np = pd.to_numeric(df['RFS'], errors='coerce').to_numpy(dtype=float)
    event_np = pd.to_numeric(df['Relapse'], errors='coerce').fillna(-1).to_numpy(dtype=int)
    valid_rows = np.isfinite(time_np) & np.isin(event_np, [0, 1])
    if valid_rows.sum() < int(args.n_splits):
        raise RuntimeError(f'Not enough valid samples for {feature_set.name} / {model_family}.')

    model_columns = unique_preserve_order(
        list(feature_set.locked_columns) + [col for block in feature_set.feature_blocks for col in block.columns]
    )
    work_df = df.loc[valid_rows, ['PatientID', 'RFS', 'Relapse'] + model_columns].reset_index(drop=True)
    patient_ids = work_df['PatientID'].astype(str).tolist()
    time_used = pd.to_numeric(work_df['RFS'], errors='coerce').to_numpy(dtype=float)
    event_used = pd.to_numeric(work_df['Relapse'], errors='coerce').to_numpy(dtype=int)
    x_df = work_df[model_columns].apply(pd.to_numeric, errors='coerce')
    y = Surv.from_arrays(event=event_used.astype(bool), time=time_used.astype(float))

    splitter = StratifiedKFold(n_splits=int(args.n_splits), shuffle=True, random_state=int(args.random_state))
    oof_risk = np.full(x_df.shape[0], np.nan, dtype=float)
    fold_rows = []
    selected_feature_rows = []

    for fold_idx, (train_idx, test_idx) in enumerate(splitter.split(x_df, event_used), start=1):
        x_train_df = x_df.iloc[train_idx].copy()
        x_test_df = x_df.iloc[test_idx].copy()
        y_train = y[train_idx]
        y_test = y[test_idx]

        nonempty = ~x_train_df.isna().all(axis=0).to_numpy()
        x_train_df = x_train_df.iloc[:, nonempty]
        x_test_df = x_test_df.iloc[:, nonempty]

        imputer = SimpleImputer(strategy='median')
        x_train = imputer.fit_transform(x_train_df)
        x_test = imputer.transform(x_test_df)
        feature_names = list(x_train_df.columns)

        stds = np.nanstd(x_train, axis=0)
        keep_var = np.isfinite(stds) & (stds > 1e-9)
        x_train = x_train[:, keep_var]
        x_test = x_test[:, keep_var]
        feature_names = [feature_names[idx] for idx in np.where(keep_var)[0]]

        name_to_index = {name: idx for idx, name in enumerate(feature_names)}
        locked_keep_names = [name for name in feature_set.locked_columns if name in name_to_index]
        locked_indices = [name_to_index[name] for name in locked_keep_names]

        selected_train_parts = []
        selected_test_parts = []
        selected_feature_names = []

        if locked_indices:
            selected_train_parts.append(x_train[:, locked_indices])
            selected_test_parts.append(x_test[:, locked_indices])
            selected_feature_names.extend(locked_keep_names)

        for block in feature_set.feature_blocks:
            block_names = [name for name in block.columns if name in name_to_index]
            if not block_names:
                continue
            block_indices = [name_to_index[name] for name in block_names]
            train_block = x_train[:, block_indices]
            test_block = x_test[:, block_indices]
            train_block, test_block, block_selected_names = select_feature_block(
                train_block,
                test_block,
                block_names,
                y_train,
                float(args.correlation_threshold),
                int(block.max_features),
            )
            if block_selected_names:
                selected_train_parts.append(train_block)
                selected_test_parts.append(test_block)
                selected_feature_names.extend(block_selected_names)

        if not selected_train_parts:
            raise RuntimeError(f'No features remained for {feature_set.name} / {model_family} in fold {fold_idx}.')

        x_train_fit = np.concatenate(selected_train_parts, axis=1)
        x_test_fit = np.concatenate(selected_test_parts, axis=1)
        x_train_fit, x_test_fit = maybe_standardize(model_family, x_train_fit, x_test_fit)

        model = instantiate_model(model_family, args, fold_idx)
        model.fit(x_train_fit, y_train)
        risk = np.asarray(model.predict(x_test_fit), dtype=float)
        oof_risk[test_idx] = risk

        fold_c = concordance_index_value(y_test['event'], y_test['time'], risk)
        fold_rows.append([fold_idx, len(train_idx), len(test_idx), int(np.sum(y_test['event'])), float(fold_c), len(selected_feature_names)])
        for feature_name in selected_feature_names:
            selected_feature_rows.append([fold_idx, feature_name])
        print(
            f'Fold {fold_idx}/{args.n_splits}: model={model_family} | feature_set={feature_set.name} | '
            f'test_n={len(test_idx)} | events={int(np.sum(y_test["event"]))} | C-index={fold_c:.3f} | '
            f'features={len(selected_feature_names)}'
        )

    c_index = concordance_index_value(event_used, time_used, oof_risk)
    fold_c_values = np.asarray([row[4] for row in fold_rows], dtype=float)
    fold_feature_counts = np.asarray([row[5] for row in fold_rows], dtype=float)

    with open(os.path.join(output_dir, 'oof_predictions.csv'), 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['PatientID', 'RFS', 'Relapse', 'risk_score'])
        for patient_id, time_val, event_val, risk_val in zip(patient_ids, time_used, event_used, oof_risk):
            writer.writerow([patient_id, float(time_val), int(event_val), float(risk_val)])

    with open(os.path.join(output_dir, 'cv_folds.csv'), 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['fold', 'train_n', 'test_n', 'test_events', 'c_index', 'selected_features'])
        writer.writerows(fold_rows)

    with open(os.path.join(output_dir, 'selected_features_by_fold.csv'), 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['fold', 'feature'])
        writer.writerows(selected_feature_rows)

    with open(os.path.join(output_dir, 'summary.txt'), 'w') as f:
        f.write(f'Model family={model_family}\n')
        f.write(f'Feature set={feature_set.name}\n')
        f.write(f'Samples={x_df.shape[0]}\n')
        f.write(f'Events={int(event_used.sum())}\n')
        f.write(f'OOF C-index={c_index:.12f}\n')
        f.write(f'Mean fold C-index={float(np.nanmean(fold_c_values)):.12f}\n')
        f.write(f'Std fold C-index={float(np.nanstd(fold_c_values)):.12f}\n')
        f.write(f'Mean selected features={float(np.nanmean(fold_feature_counts)):.3f}\n')

    print(f'Completed: model={model_family} | feature_set={feature_set.name} | OOF C-index={c_index:.3f}')
    return BenchmarkResult(
        model_family=model_family,
        feature_set=feature_set.name,
        title=feature_set.title,
        samples=int(x_df.shape[0]),
        events=int(event_used.sum()),
        oof_c_index=float(c_index),
        mean_fold_c_index=float(np.nanmean(fold_c_values)),
        std_fold_c_index=float(np.nanstd(fold_c_values)),
        mean_selected_features=float(np.nanmean(fold_feature_counts)),
        output_dir=output_dir,
    )


def main() -> None:
    args = parse_args()
    model_families = parse_csv_items(args.models, MODEL_CHOICES)
    summary_families = parse_csv_items(args.atlas_summary_families, SUMMARY_FAMILY_CHOICES)

    df = merge_tumor_and_atlas_tables(args.tumor_csv, args.atlas_csv)
    df, clinical_cols, tumor_cols, atlas_cols = collect_model_columns(df, args.atlas_feature_mode, summary_families)
    if not clinical_cols:
        raise RuntimeError('No clinical columns found.')
    if not tumor_cols:
        raise RuntimeError('No tumor imaging columns found.')
    if not atlas_cols:
        raise RuntimeError('No atlas biomarker columns found.')

    print(
        f'Benchmark setup: models={model_families} | atlas_families={summary_families} | '
        f'tumor_cols={len(tumor_cols)} | atlas_cols={len(atlas_cols)}'
    )

    feature_sets = build_feature_sets(clinical_cols, tumor_cols, atlas_cols, args)
    os.makedirs(args.output_dir, exist_ok=True)
    results: List[BenchmarkResult] = []
    for model_family in model_families:
        for feature_set in feature_sets:
            result_dir = os.path.join(args.output_dir, model_family, feature_set.name)
            results.append(cross_validated_benchmark(df, feature_set, model_family, result_dir, args))

    results_csv = os.path.join(args.output_dir, 'model_benchmark_comparison.csv')
    with open(results_csv, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            'model_family', 'feature_set', 'title', 'samples', 'events', 'oof_c_index', 'mean_fold_c_index',
            'std_fold_c_index', 'mean_selected_features', 'output_dir'
        ])
        for result in results:
            writer.writerow([
                result.model_family,
                result.feature_set,
                result.title,
                result.samples,
                result.events,
                result.oof_c_index,
                result.mean_fold_c_index,
                result.std_fold_c_index,
                result.mean_selected_features,
                result.output_dir,
            ])

    best = max(results, key=lambda item: item.oof_c_index)
    summary_txt = os.path.join(args.output_dir, 'model_benchmark_summary.txt')
    with open(summary_txt, 'w') as f:
        f.write('Matched OOF survival-model benchmark\n')
        f.write(f'Models={model_families}\n')
        f.write(f'Atlas feature mode={args.atlas_feature_mode}\n')
        f.write(f'Atlas summary families={summary_families}\n')
        f.write(f'Max tumor features={args.max_tumor_features}\n')
        f.write(f'Max atlas features={args.max_atlas_features}\n\n')
        for result in sorted(results, key=lambda item: (item.model_family, -item.oof_c_index, item.feature_set)):
            f.write(
                f'{result.model_family} | {result.feature_set}: OOF C-index={result.oof_c_index:.12f} | '
                f'mean fold={result.mean_fold_c_index:.12f}±{result.std_fold_c_index:.12f} | '
                f'mean selected features={result.mean_selected_features:.2f}\n'
            )
        f.write(
            f'\nBest result: {best.model_family} | {best.feature_set} | '
            f'OOF C-index={best.oof_c_index:.12f}\n'
        )

    print(f'Saved benchmark comparison: {results_csv}')
    print(f'Best result: {best.model_family} | {best.feature_set} | OOF C-index={best.oof_c_index:.3f}')


if __name__ == '__main__':
    main()
