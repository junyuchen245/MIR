import argparse
import os

from rsf_survival_common import (
    FeatureBlockSpec,
    ModelSpec,
    get_clinical_columns,
    get_imaging_columns,
    load_case_feature_table,
    run_model_suite,
)


THIS_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_FEATURE_CSV = os.path.join(THIS_DIR, 'population_stats_tumor_radiomics', 'tumor_radiomics_case_features.csv')
DEFAULT_OUTPUT_DIR = os.path.join(THIS_DIR, 'population_stats_tumor_rsf')


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='RSF survival comparison for tumor-only imaging biomarkers: clinical only vs clinical + tumor imaging.'
    )
    parser.add_argument('--feature-csv', default=DEFAULT_FEATURE_CSV)
    parser.add_argument('--output-dir', default=DEFAULT_OUTPUT_DIR)
    parser.add_argument('--n-splits', type=int, default=10)
    parser.add_argument('--n-estimators', type=int, default=500)
    parser.add_argument('--min-samples-leaf', type=int, default=5)
    parser.add_argument('--max-tumor-features', type=int, default=50, help='Maximum number of tumor/radiomics features kept per fold after filtering; 0 disables capping.')
    parser.add_argument('--correlation-threshold', type=float, default=0.90)
    parser.add_argument('--random-state', type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    df = load_case_feature_table(args.feature_csv)
    clinical_cols = get_clinical_columns(df)
    tumor_cols = get_imaging_columns(df)
    if not clinical_cols:
        raise RuntimeError('No clinical columns were found in the tumor case feature table.')
    if not tumor_cols:
        raise RuntimeError('No tumor imaging columns were found in the tumor case feature table.')

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
        suite_title='Tumor RSF comparison: clinical baseline vs clinical + tumor imaging',
    )


if __name__ == '__main__':
    main()
