#!/usr/bin/env python3
import argparse
import csv
import math
import re
import statistics
from collections import Counter, defaultdict
from datetime import datetime
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

DEFAULT_CSV = "/scratch2/jchen/DATA/AutoPET/Metadata.csv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Summarize the AutoPET metadata CSV into a paper-style cohort table. "
            "Defaults to one baseline record per subject using the earliest study date."
        )
    )
    parser.add_argument(
        "--input-csv",
        default=DEFAULT_CSV,
        help="Path to the AutoPET Metadata.csv file.",
    )
    parser.add_argument(
        "--unit",
        choices=["subject", "study", "series"],
        default="subject",
        help="Summarize unique subjects, unique studies, or raw series rows.",
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


def parse_age_years(value: object) -> float:
    text = clean_text(value)
    match = re.match(r"^(\d+)Y$", text)
    if match:
        return float(match.group(1))
    try:
        return float(text)
    except Exception:
        return math.nan


def parse_date(value: object) -> Optional[datetime]:
    text = clean_text(value)
    if not text:
        return None
    for fmt in ("%m/%d/%Y", "%m/%d/%y", "%Y-%m-%d"):
        try:
            return datetime.strptime(text, fmt)
        except Exception:
            pass
    return None


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


def normalize_sex(value: object) -> str:
    text = clean_text(value).upper()
    if text == "M":
        return "Male"
    if text == "F":
        return "Female"
    return "Missing"


def normalize_diagnosis(value: object) -> str:
    text = clean_text(value).upper()
    if not text:
        return "Missing"
    mapping = {
        "NEGATIVE": "Negative",
        "MELANOMA": "Melanoma",
        "LUNG_CANCER": "Lung cancer",
        "LYMPHOMA": "Lymphoma",
    }
    return mapping.get(text, text.replace("_", " ").title())


def load_rows(path: str) -> List[Dict[str, object]]:
    with open(path, newline="") as handle:
        return list(csv.DictReader(handle))


def build_study_rows(series_rows: Sequence[Dict[str, object]]) -> List[Dict[str, object]]:
    studies: Dict[Tuple[str, str], List[Dict[str, object]]] = defaultdict(list)
    for row in series_rows:
        studies[(clean_text(row.get("Subject ID")), clean_text(row.get("Study UID")))].append(row)

    out: List[Dict[str, object]] = []
    for (_, _), group in studies.items():
        representative = dict(group[0])
        representative["_series_count"] = len(group)
        representative["_modalities"] = "+".join(sorted({clean_text(item.get("Modality")) for item in group if clean_text(item.get("Modality"))}))
        representative["_manufacturers"] = "+".join(sorted({clean_text(item.get("Manufacturer")) for item in group if clean_text(item.get("Manufacturer"))}))
        representative["_scanner_manufacturers"] = "+".join(
            sorted(
                {
                    clean_text(item.get("Manufacturer"))
                    for item in group
                    if clean_text(item.get("Manufacturer")) and clean_text(item.get("Modality")) != "SEG"
                }
            )
        )
        representative["_study_date"] = parse_date(group[0].get("Study Date"))
        out.append(representative)
    return out


def build_subject_rows(study_rows: Sequence[Dict[str, object]], study_choice: str) -> Tuple[List[Dict[str, object]], int]:
    subjects: Dict[str, List[Dict[str, object]]] = defaultdict(list)
    for row in study_rows:
        subjects[clean_text(row.get("Subject ID"))].append(row)

    selected: List[Dict[str, object]] = []
    diagnosis_transition_subjects = 0
    reverse = study_choice == "latest"

    for subject_id, rows in subjects.items():
        diagnoses = {normalize_diagnosis(row.get("diagnosis")) for row in rows}
        if len(diagnoses) > 1:
            diagnosis_transition_subjects += 1
        sorted_rows = sorted(
            rows,
            key=lambda row: (row.get("_study_date") is None, row.get("_study_date")),
            reverse=reverse,
        )
        representative = dict(sorted_rows[0])
        representative["_study_count"] = len(rows)
        representative["_subject_modalities"] = "+".join(sorted({clean_text(item.get("_modalities")) for item in rows if clean_text(item.get("_modalities"))}))
        representative["_subject_manufacturers"] = "+".join(sorted({clean_text(item.get("_manufacturers")) for item in rows if clean_text(item.get("_manufacturers"))}))
        representative["_subject_scanner_manufacturers"] = "+".join(sorted({clean_text(item.get("_scanner_manufacturers")) for item in rows if clean_text(item.get("_scanner_manufacturers"))}))
        representative["_all_diagnoses"] = "; ".join(sorted(diagnoses))
        selected.append(representative)
    return selected, diagnosis_transition_subjects


def summarize(rows: Sequence[Dict[str, object]], unit: str, study_choice: str, diagnosis_transition_subjects: int) -> Tuple[List[Tuple[str, str]], List[str]]:
    ages = [parse_age_years(row.get("age")) for row in rows]
    sexes = Counter(normalize_sex(row.get("sex")) for row in rows)
    diagnoses = Counter(normalize_diagnosis(row.get("diagnosis")) for row in rows)
    modality_counter = Counter()
    manufacturer_counter = Counter()
    scanner_manufacturer_counter = Counter()

    if unit == "series":
        modality_counter.update(clean_text(row.get("Modality")) or "Missing" for row in rows)
        manufacturer_counter.update(clean_text(row.get("Manufacturer")) or "Missing" for row in rows)
        study_count_values: List[float] = []
    elif unit == "study":
        modality_counter.update(clean_text(row.get("_modalities")) or "Missing" for row in rows)
        manufacturer_counter.update(clean_text(row.get("_manufacturers")) or "Missing" for row in rows)
        scanner_manufacturer_counter.update(clean_text(row.get("_scanner_manufacturers")) or "Missing" for row in rows)
        study_count_values = [float(row.get("_series_count", math.nan)) for row in rows]
    else:
        modality_counter.update(clean_text(row.get("_subject_modalities")) or "Missing" for row in rows)
        manufacturer_counter.update(clean_text(row.get("_subject_manufacturers")) or "Missing" for row in rows)
        scanner_manufacturer_counter.update(clean_text(row.get("_subject_scanner_manufacturers")) or "Missing" for row in rows)
        study_count_values = [float(row.get("_study_count", math.nan)) for row in rows]

    table: List[Tuple[str, str]] = []
    label = {
        "subject": "AutoPET cohort (subjects)",
        "study": "AutoPET cohort (studies)",
        "series": "AutoPET cohort (series)",
    }[unit]
    table.append((label, str(len(rows))))
    table.append(("Age (y) (mean ± SD)", mean_sd(ages)))
    table.append(("Sex", ""))
    for key in ["Male", "Female", "Missing"]:
        table.append((f"  {key}", str(sexes.get(key, 0))))
    table.append(("Diagnosis", ""))
    for key in ["Negative", "Melanoma", "Lung cancer", "Lymphoma", "Missing"]:
        table.append((f"  {key}", str(diagnoses.get(key, 0))))

    if unit != "series":
        descriptor = "Studies per subject" if unit == "subject" else "Series per study"
        table.append((f"{descriptor} median (range)", median_range(study_count_values, decimals=0)))

    table.append(("Modality coverage", ""))
    for key, count in modality_counter.most_common():
        table.append((f"  {key}", str(count)))

    table.append(("Manufacturer", ""))
    for key, count in manufacturer_counter.most_common():
        table.append((f"  {key or 'Missing'}", str(count)))
    if unit != "series":
        table.append(("Scanner manufacturer (CT/PT only)", ""))
        for key, count in scanner_manufacturer_counter.most_common():
            table.append((f"  {key or 'Missing'}", str(count)))

    notes: List[str] = []
    if unit == "subject":
        notes.append(f"Subject mode keeps one representative study per subject using the {study_choice} study date.")
        notes.append(f"{diagnosis_transition_subjects} subjects have diagnosis labels that change across repeated studies, most commonly from a cancer label to Negative.")
        notes.append("Diagnosis, age, and sex are taken from the representative study row for each subject.")
    elif unit == "study":
        notes.append("Study mode collapses repeated CT/PT/SEG series into one row per Study UID.")
    else:
        notes.append("Series mode reports raw DICOM series rows from the metadata export.")
    notes.append("AutoPET metadata only provide age, sex, diagnosis, and imaging metadata; no formal treatment or outcome fields are available in this CSV.")
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
    series_rows = load_rows(args.input_csv)

    if args.unit == "series":
        rows = series_rows
        diagnosis_transition_subjects = 0
    else:
        study_rows = build_study_rows(series_rows)
        if args.unit == "study":
            rows = study_rows
            diagnosis_transition_subjects = 0
        else:
            rows, diagnosis_transition_subjects = build_subject_rows(study_rows, args.study_choice)

    table, notes = summarize(rows, args.unit, args.study_choice, diagnosis_transition_subjects)

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
