import argparse
import os

from rsf_survival_common import (
    ATLAS_PREFIX,
    FeatureBlockSpec,
    ModelSpec,
    add_atlas_pet_summary_features,
    get_clinical_columns,
    merge_tumor_and_atlas_tables,
    run_model_suite,
)


THIS_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_TUMOR_CSV = os.path.join(THIS_DIR, 'population_stats_tumor_radiomics', 'tumor_radiomics_case_features.csv')
DEFAULT_ATLAS_CSV = os.path.join(THIS_DIR, 'population_stats_157lbls', 'imaging_case_features.csv')
DEFAULT_OUTPUT_DIR = os.path.join(THIS_DIR, 'population_stats_atlas_rsf')
SUMMARY_FAMILY_CHOICES = ('burden', 'overlap', 'z')


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='RSF survival comparison to test incremental atlas value beyond clinical and tumor imaging predictors.'
    )
    parser.add_argument('--tumor-csv', default=DEFAULT_TUMOR_CSV)
    parser.add_argument('--atlas-csv', default=DEFAULT_ATLAS_CSV)
    parser.add_argument('--output-dir', default=DEFAULT_OUTPUT_DIR)
    parser.add_argument('--n-splits', type=int, default=10)
    parser.add_argument('--n-estimators', type=int, default=500)
    parser.add_argument('--min-samples-leaf', type=int, default=5)
    parser.add_argument('--max-tumor-features', type=int, default=100, help='Maximum number of tumor/radiomics features kept per fold after filtering; 0 disables capping.')
    parser.add_argument('--max-atlas-features', type=int, default=4, help='Maximum number of atlas biomarker features kept per fold after filtering; 0 disables capping.')
    parser.add_argument('--atlas-feature-mode', choices=['summary', 'raw'], default='summary', help='Use compact atlas summary features (recommended) or the raw atlas region feature set.')
    parser.add_argument(
        '--atlas-summary-families',
        default='overlap',
        help='Comma-separated atlas summary families to include when --atlas-feature-mode=summary. Choices: burden, overlap, z.',
    )
    parser.add_argument('--correlation-threshold', type=float, default=0.90)
    parser.add_argument('--random-state', type=int, default=42)
    return parser.parse_args()


def parse_summary_families(raw_value: str):
    items = [item.strip().lower() for item in str(raw_value).split(',') if item.strip()]
    if not items:
        raise ValueError('At least one atlas summary family must be provided.')
    bad = [item for item in items if item not in SUMMARY_FAMILY_CHOICES]
    if bad:
        raise ValueError(
            f'Unsupported atlas summary family/families: {bad}. Expected subset of {SUMMARY_FAMILY_CHOICES}.'
        )
    seen = set()
    ordered = []
    for item in items:
        if item not in seen:
            ordered.append(item)
            seen.add(item)
    return ordered


def main() -> None:
    args = parse_args()
    summary_families = parse_summary_families(args.atlas_summary_families)
    df = merge_tumor_and_atlas_tables(args.tumor_csv, args.atlas_csv)
    clinical_cols = get_clinical_columns(df)
    tumor_cols = [col for col in df.columns if col.startswith(('tumor_', 'pet_original_', 'ct_original_'))]

    def collect_prefixed_family(prefix: str):
        merged_prefix = f'{ATLAS_PREFIX}{prefix}'
        return [col for col in df.columns if col.startswith((merged_prefix, prefix))]

    atlas_burden_cols = collect_prefixed_family('atlas_burden_')
    atlas_overlap_cols = collect_prefixed_family('atlas_overlap_')
    atlas_z_cols = collect_prefixed_family('atlas_z_')
    if args.atlas_feature_mode == 'summary':
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
            print(
                'Atlas summary mode: no atlas burden/overlap/z columns found in merged table; '
                f'falling back to legacy atlas PET summaries ({len(atlas_summary_cols)} columns).'
            )
        else:
            print(
                'Atlas summary mode: '
                f'families={summary_families} | burden={len(atlas_burden_cols)} | '
                f'overlap={len(atlas_overlap_cols)} | z={len(atlas_z_cols)} | total={len(atlas_cols)}'
            )
    else:
        atlas_cols = [
            col for col in df.columns
            if col.startswith(ATLAS_PREFIX) or col.startswith('atlas_overlap_') or col.startswith('atlas_burden_') or col.startswith('atlas_z_')
        ]
    if not clinical_cols:
        raise RuntimeError('No clinical columns were found in the merged tumor+atlas case table.')
    if not tumor_cols:
        raise RuntimeError('No tumor imaging columns were found in the merged tumor+atlas case table.')
    if not atlas_cols:
        raise RuntimeError('No atlas-only biomarker columns were found in the merged tumor+atlas case table.')

    model_specs = [
        ModelSpec(
            name='clinical_only',
            title='Clinical only',
            locked_columns=list(clinical_cols),
        ),
        ModelSpec(
            name='clinical_plus_tumor',
            title='Clinical + tumor imaging',
            locked_columns=list(clinical_cols),
            feature_blocks=[
                FeatureBlockSpec(
                    name='tumor',
                    columns=list(tumor_cols),
                    max_features=int(args.max_tumor_features),
                )
            ],
        ),
        ModelSpec(
            name='clinical_plus_atlas',
            title='Clinical + atlas biomarkers',
            locked_columns=list(clinical_cols),
            feature_blocks=[
                FeatureBlockSpec(
                    name='atlas',
                    columns=list(atlas_cols),
                    max_features=int(args.max_atlas_features),
                )
            ],
        ),
        ModelSpec(
            name='clinical_plus_tumor_plus_atlas',
            title='Clinical + tumor imaging + atlas biomarkers',
            locked_columns=list(clinical_cols),
            feature_blocks=[
                FeatureBlockSpec(
                    name='tumor',
                    columns=list(tumor_cols),
                    max_features=int(args.max_tumor_features),
                ),
                FeatureBlockSpec(
                    name='atlas',
                    columns=list(atlas_cols),
                    max_features=int(args.max_atlas_features),
                ),
            ],
        ),
    ]

    run_model_suite(
        df=df,
        model_specs=model_specs,
        output_dir=args.output_dir,
        n_splits=args.n_splits,
        n_estimators=args.n_estimators,
        min_samples_leaf=args.min_samples_leaf,
        correlation_threshold=args.correlation_threshold,
        random_state=args.random_state,
        suite_title='Atlas RSF comparison: incremental value beyond clinical and tumor imaging',
    )


if __name__ == '__main__':
    main()
