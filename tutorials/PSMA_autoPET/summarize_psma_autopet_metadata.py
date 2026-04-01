#!/usr/bin/env python3
import argparse
import csv
import math
import statistics
from collections import Counter, defaultdict
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

DEFAULT_CSV = "/scratch2/jchen/DATA/PSMA_autoPET/psma_metadata.csv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Summarize the PSMA AutoPET metadata CSV into a paper-style cohort table. "
            "Defaults to one baseline row per subject using the earliest study date."
        )
    )
    parser.add_argument(
        "--input-csv",
        default=DEFAULT_CSV,
        help="Path to the PSMA AutoPET metadata CSV.",
    )
    parser.add_argument(
        "--unit",
        choices=["subject", "study"],
        default="subject",
        help="Summarize unique subjects or all study rows.",
    )
    parser.add_argument(
        "--study-choice",
        choices=["earliest", "latest"],
        default="earliest",
        help="Which study to keep when summarizing at the subject level.",
    )
    parser.add_argument(
        "--output-format",
        choices=["text", "markdown", "csv"],
        default="text",
        help="Output format.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Optional output file path.",
    )
    return parser.parse_args()


def clean_text(value: object) -> str:
    if value is None:
        return ""
    return str(value).strip()


def to_float(value: object) -> float:
    text = clean_text(value)
    if not text:
        return math.nan
    try:
        return float(text)
    except Exception:
        return math.nan


def valid_numbers(values: Iterable[float]) -> List[float]:
    return [value for value in values if not math.isnan(value)]


def mean_sd(values: Iterable[float], decimals: int = 2) -> str:
    nums = valid_numbers(values)
    if not nums:
        return "NA"
    avg = statistics.mean(nums)
    sd = statistics.stdev(nums) if len(nums) > 1 else 0.0
    return f"{avg:.{decimals}f} ± {sd:.{decimals}f}"


def median_range(values: Iterable[float], decimals: int = 2) -> str:
    nums = valid_numbers(values)
    if not nums:
        return "NA"
    return f"{statistics.median(nums):.{decimals}f} ({min(nums):.{decimals}f}–{max(nums):.{decimals}f})"


def normalize_radionuclide(value: object) -> str:
    text = clean_text(value)
    return text if text else "Missing"


def normalize_contrast(value: object) -> str:
    text = clean_text(value).lower()
    if text == "yes":
        return "Contrast"
    if text == "no":
        return "No contrast"
    return "Missing"


def normalize_model(value: object) -> str:
    text = clean_text(value)
    return text if text else "Missing"


def load_rows(path: str) -> List[Dict[str, str]]:
    with open(path, newline="") as handle:
        return list(csv.DictReader(handle))


def build_subject_rows(study_rows: Sequence[Dict[str, str]], study_choice: str) -> Tuple[List[Dict[str, str]], Counter]:
    subjects: Dict[str, List[Dict[str, str]]] = defaultdict(list)
    for row in study_rows:
        subjects[clean_text(row.get("Subject ID"))].append(row)

    reverse = study_choice == "latest"
    selected: List[Dict[str, str]] = []
    changes = Counter()
    for subject_id, rows in subjects.items():
        rads = {normalize_radionuclide(row.get("pet_radionuclide")) for row in rows}
        contrasts = {normalize_contrast(row.get("ct_contrast_agent")) for row in rows}
        models = {normalize_model(row.get("manufacturer_model_name")) for row in rows}
        if len(rads) > 1:
            changes["radionuclide"] += 1
        if len(contrasts) > 1:
            changes["contrast"] += 1
        if len(models) > 1:
            changes["model"] += 1
        sorted_rows = sorted(rows, key=lambda row: clean_text(row.get("Study Date")), reverse=reverse)
        representative = dict(sorted_rows[0])
        representative["_study_count"] = len(rows)
        representative["_all_radionuclides"] = "; ".join(sorted(rads))
        representative["_all_contrast"] = "; ".join(sorted(contrasts))
        representative["_all_models"] = "; ".join(sorted(models))
        representative["_first_year"] = clean_text(sorted(rows, key=lambda row: clean_text(row.get("Study Date")))[0].get("Study Date"))[:4]
        selected.append(representative)
    return selected, changes


def summarize(rows: Sequence[Dict[str, str]], unit: str, study_choice: str, changes: Counter) -> Tuple[List[Tuple[str, str]], List[str]]:
    ages = [to_float(row.get("age")) for row in rows]
    radionuclide_counts = Counter(normalize_radionuclide(row.get("pet_radionuclide")) for row in rows)
    contrast_counts = Counter(normalize_contrast(row.get("ct_contrast_agent")) for row in rows)
    model_counts = Counter(normalize_model(row.get("manufacturer_model_name")) for row in rows)
    year_counts = Counter(clean_text(row.get("Study Date"))[:4] or "Missing" for row in rows)

    table: List[Tuple[str, str]] = []
    table.append((f"PSMA AutoPET cohort ({'subjects' if unit == 'subject' else 'studies'})", str(len(rows))))
    table.append(("Age (y) (mean ± SD)", mean_sd(ages)))

    if unit == "subject":
        studies_per_subject = [float(row.get("_study_count", math.nan)) for row in rows]
        table.append(("Studies per subject median (range)", median_range(studies_per_subject, decimals=0)))

    table.append(("PET radionuclide", ""))
    for label in ["18F", "68Ga", "Missing"]:
        table.append((f"  {label}", str(radionuclide_counts.get(label, 0))))

    table.append(("CT contrast agent", ""))
    for label in ["Contrast", "No contrast", "Missing"]:
        table.append((f"  {label}", str(contrast_counts.get(label, 0))))

    table.append(("Scanner model", ""))
    for label, count in model_counts.most_common():
        table.append((f"  {label}", str(count)))

    table.append(("Study year", ""))
    for label, count in sorted(year_counts.items(), key=lambda item: (item[0] == "Missing", item[0])):
        table.append((f"  {label}", str(count)))

    notes: List[str] = []
    if unit == "subject":
        notes.append(f"Subject mode keeps one representative study per subject using the {study_choice} study date.")
        notes.append(f"{changes.get('radionuclide', 0)} subjects change radionuclide across follow-up studies.")
        notes.append(f"{changes.get('contrast', 0)} subjects change CT contrast usage across studies.")
        notes.append(f"{changes.get('model', 0)} subjects change scanner model across studies.")
    else:
        notes.append("Study mode keeps every study row from the metadata CSV.")
    notes.append("This metadata file contains age and acquisition metadata only; no sex, disease stage, therapy, or outcome fields are present.")
    return table, notes


def render_text(table: Sequence[Tuple[str, str]], notes: Sequence[str]) -> str:
    width = max(len(label) for label, _ in table)
    lines: List[str] = []
    for label, value in table:
        if value:
            lines.append(f"{label.ljust(width)}  {value}")
        else:
            lines.append(label)
    if notes:
        lines.append("")
        lines.append("Notes")
        for note in notes:
            lines.append(f"- {note}")
    return "\n".join(lines)


def render_markdown(table: Sequence[Tuple[str, str]], notes: Sequence[str]) -> str:
    lines = ["| Variable | Value |", "| --- | --- |"]
    for label, value in table:
        lines.append(f"| {label} | {value} |")
    if notes:
        lines.append("")
        lines.append("**Notes**")
        for note in notes:
            lines.append(f"- {note}")
    return "\n".join(lines)


def write_csv(table: Sequence[Tuple[str, str]], output_path: str) -> None:
    with open(output_path, "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["Variable", "Value"])
        writer.writerows(table)


def main() -> None:
    args = parse_args()
    study_rows = load_rows(args.input_csv)
    if args.unit == "study":
        rows = study_rows
        changes = Counter()
    else:
        rows, changes = build_subject_rows(study_rows, args.study_choice)

    table, notes = summarize(rows, args.unit, args.study_choice, changes)

    if args.output_format == "csv":
        if not args.output:
            raise ValueError("--output is required when --output-format csv is used.")
        write_csv(table, args.output)
        return

    rendered = render_markdown(table, notes) if args.output_format == "markdown" else render_text(table, notes)
    if args.output:
        with open(args.output, "w") as handle:
            handle.write(rendered + "\n")
    else:
        print(rendered)


if __name__ == "__main__":
    main()
