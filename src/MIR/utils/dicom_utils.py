"""DICOM header inspection utilities.

Scan a directory of DICOM files, group them by series, and print a summary of
the key header fields (patient/study/series, geometry, modality-specific
acquisition parameters). Optionally dump the full header of one representative
file per series.

Command line usage::

    python -m MIR.utils.dicom_utils /path/to/dicom_dir
    python -m MIR.utils.dicom_utils /path/to/dicom_dir --full
    python -m MIR.utils.dicom_utils /path/to/dicom_dir --tags PatientWeight 0x00281053
    python -m MIR.utils.dicom_utils /path/to/dicom_dir --hide-phi

Requires ``pydicom`` (``pip install pydicom``).
"""

import argparse
import os
import sys
from collections import OrderedDict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, TextIO, Tuple, Union


# Header fields shown in the per-series summary, grouped by section.
SUMMARY_FIELDS = OrderedDict(
    [
        (
            'Patient',
            [
                'PatientName',
                'PatientID',
                'PatientBirthDate',
                'PatientSex',
                'PatientAge',
                'PatientWeight',
                'PatientSize',
            ],
        ),
        (
            'Study',
            ['StudyInstanceUID', 'StudyDate', 'StudyTime', 'StudyDescription', 'AccessionNumber', 'InstitutionName'],
        ),
        (
            'Series',
            [
                'SeriesInstanceUID',
                'SeriesNumber',
                'SeriesDescription',
                'Modality',
                'SOPClassUID',
                'SeriesDate',
                'SeriesTime',
                'BodyPartExamined',
                'Manufacturer',
                'ManufacturerModelName',
                'FrameOfReferenceUID',
            ],
        ),
        (
            'Geometry',
            [
                'Rows',
                'Columns',
                'NumberOfFrames',
                'PixelSpacing',
                'SliceThickness',
                'SpacingBetweenSlices',
                'ImageOrientationPatient',
                'PatientPosition',
                'BitsStored',
                'PixelRepresentation',
                'PhotometricInterpretation',
                'RescaleSlope',
                'RescaleIntercept',
                'RescaleType',
            ],
        ),
        ('CT', ['KVP', 'XRayTubeCurrent', 'Exposure', 'ConvolutionKernel', 'ReconstructionDiameter', 'CTDIvol']),
        (
            'MR',
            [
                'MagneticFieldStrength',
                'SequenceName',
                'ScanningSequence',
                'RepetitionTime',
                'EchoTime',
                'InversionTime',
                'FlipAngle',
                'ReceiveCoilName',
            ],
        ),
        (
            'PET',
            [
                'Units',
                'DecayCorrection',
                'CorrectedImage',
                'AttenuationCorrectionMethod',
                'ReconstructionMethod',
                'AcquisitionTime',
                'FrameReferenceTime',
                'ActualFrameDuration',
                'DecayFactor',
                'DoseCalibrationFactor',
            ],
        ),
    ]
)

# Attributes treated as protected health information when ``hide_phi`` is set.
PHI_FIELDS = {
    'PatientName',
    'PatientID',
    'PatientBirthDate',
    'OtherPatientIDs',
    'OtherPatientNames',
    'PatientAddress',
    'PatientTelephoneNumbers',
    'AccessionNumber',
    'InstitutionName',
    'InstitutionAddress',
    'ReferringPhysicianName',
    'PerformingPhysicianName',
    'OperatorsName',
    'StationName',
    'StudyID',
}
PHI_MASK = '<hidden>'


def _require_pydicom():
    try:
        import pydicom
    except ImportError as e:
        raise ImportError('dicom_utils requires pydicom: pip install pydicom') from e
    return pydicom


def _format_value(value, max_len=120):
    """Format a DICOM element value as a single-line string."""
    if value is None:
        return ''
    if isinstance(value, (bytes, bytearray)):
        return f'<{len(value)} bytes>'
    if hasattr(value, '__iter__') and not isinstance(value, str):
        try:
            value = '[' + ', '.join(str(v) for v in value) + ']'
        except TypeError:
            value = str(value)
    s = str(value).replace('\n', ' ').replace('\r', ' ')
    return s if len(s) <= max_len else s[: max_len - 3] + '...'


def _parse_tag(tag):
    """Convert a keyword or hex tag string (e.g. '0x00281053', '0028,1053') to a lookup key."""
    t = tag.strip().replace('(', '').replace(')', '').replace(' ', '')
    if ',' in t:
        g, e = t.split(',')
        return int(g, 16) << 16 | int(e, 16)
    if t.lower().startswith('0x'):
        return int(t, 16)
    if len(t) == 8 and all(c in '0123456789abcdefABCDEF' for c in t):
        return int(t, 16)
    return t


def _get_element_value(ds, tag):
    """Return (name, value) for a keyword or numeric tag, or (name, None) if absent."""
    key = _parse_tag(tag)
    if isinstance(key, str):
        return key, getattr(ds, key, None) if key in ds else None
    if key in ds:
        elem = ds[key]
        return f'{elem.tag} {elem.keyword or elem.name}', elem.value
    return f'({key >> 16:04X},{key & 0xFFFF:04X})', None


def scan_dicom_dir(dicom_dir: Union[str, os.PathLike], recursive: bool = True) -> Dict[str, List[Tuple[Path, Any]]]:
    """Read the headers of all DICOM files in a directory and group them by series.

    Non-DICOM files (unreadable, or without SOPClassUID and SeriesInstanceUID) and hidden files (names starting
    with ``'.'``) are skipped silently. Pixel data is not loaded.

    Args:
        dicom_dir: Directory containing DICOM files (or a single DICOM file).
        recursive: Whether to descend into subdirectories.

    Returns:
        OrderedDict mapping SeriesInstanceUID (``'UNKNOWN'`` if absent) to a list of ``(path, dataset)``
        tuples, sorted by InstanceNumber, then ImagePositionPatient z, within each series.

    Raises:
        FileNotFoundError: If ``dicom_dir`` does not exist.
        ImportError: If pydicom is not installed.

    """
    pydicom = _require_pydicom()
    dicom_dir = Path(dicom_dir)
    if dicom_dir.is_file():
        files = [dicom_dir]
    elif dicom_dir.is_dir():
        it = dicom_dir.rglob('*') if recursive else dicom_dir.iterdir()
        # Skip macOS AppleDouble ('._*') and other hidden files.
        files = sorted(p for p in it if p.is_file() and not p.name.startswith('.'))
    else:
        raise FileNotFoundError(f'{dicom_dir} does not exist')

    series = OrderedDict()
    for f in files:
        try:
            ds = pydicom.dcmread(str(f), stop_before_pixels=True)
        except Exception:
            continue
        if 'SOPClassUID' not in ds and 'SeriesInstanceUID' not in ds:
            continue
        uid = str(getattr(ds, 'SeriesInstanceUID', 'UNKNOWN'))
        series.setdefault(uid, []).append((f, ds))

    def _sort_key(item):
        ds = item[1]
        ipp = getattr(ds, 'ImagePositionPatient', None)
        z = float(ipp[2]) if ipp is not None and len(ipp) == 3 else 0.0
        inst = getattr(ds, 'InstanceNumber', None)
        return (int(inst) if inst is not None else 0, z)

    for uid in series:
        series[uid].sort(key=_sort_key)
    return series


def summarize_series(datasets: Sequence[Tuple[Any, Any]], hide_phi: bool = False) -> Dict[str, Dict[str, str]]:
    """Collect the summary header fields for one series.

    Args:
        datasets: List of ``(path, dataset)`` tuples from :func:`scan_dicom_dir`.
        hide_phi: Mask patient-identifying fields.

    Fields are read from the first dataset; PET radiopharmaceutical fields come from its
    RadiopharmaceuticalInformationSequence. A ``'Derived'`` section (file count, first image position, slice z range,
    median slice gap, example file) is computed from all datasets.

    Returns:
        OrderedDict mapping section name to an OrderedDict of field -> string value.
        Sections whose fields are all absent are omitted.

    """
    ds = datasets[0][1]
    summary = OrderedDict()
    for section, fields in SUMMARY_FIELDS.items():
        entries = OrderedDict()
        for k in fields:
            if k in ds:
                entries[k] = PHI_MASK if hide_phi and k in PHI_FIELDS else _format_value(ds.data_element(k).value)
        if entries:
            summary[section] = entries

    # PET radiopharmaceutical info lives in a nested sequence.
    rp_seq = getattr(ds, 'RadiopharmaceuticalInformationSequence', None)
    if rp_seq:
        rp = rp_seq[0]
        entries = summary.setdefault('PET', OrderedDict())
        for k in [
            'Radiopharmaceutical',
            'RadionuclideTotalDose',
            'RadionuclideHalfLife',
            'RadiopharmaceuticalStartTime',
            'RadiopharmaceuticalStartDateTime',
        ]:
            if k in rp:
                entries[k] = _format_value(rp.data_element(k).value)
        code_seq = getattr(rp, 'RadionuclideCodeSequence', None)
        if code_seq and 'CodeMeaning' in code_seq[0]:
            entries['Radionuclide'] = str(code_seq[0].CodeMeaning)

    # Series-level information derived from all files.
    derived = OrderedDict()
    derived['NumFiles'] = str(len(datasets))
    positions = [d.ImagePositionPatient for _, d in datasets if 'ImagePositionPatient' in d]
    if positions:
        zs = sorted(float(p[2]) for p in positions)
        derived['FirstImagePosition'] = _format_value(positions[0])
        derived['SliceZRange'] = f'{zs[0]:.3f} .. {zs[-1]:.3f}'
        if len(zs) > 1:
            gaps = [b - a for a, b in zip(zs[:-1], zs[1:])]
            derived['MedianSliceGap'] = f'{sorted(gaps)[len(gaps) // 2]:.4f}'
    derived['ExampleFile'] = str(datasets[0][0])
    summary['Derived'] = derived
    return summary


def full_header_lines(ds: Any, hide_phi: bool = False) -> List[str]:
    """Return the full header of a dataset as a list of text lines.

    Args:
        ds: pydicom Dataset.
        hide_phi: Mask patient-identifying fields.

    Returns:
        List of strings, one per element (nested sequences are indented). File meta information, if present,
        is listed first.

    """
    lines = []

    def _walk(dataset, indent):
        for elem in dataset:
            pad = '  ' * indent
            if elem.VR == 'SQ':
                lines.append(f'{pad}{elem.tag} {elem.name}: Sequence of {len(elem.value)} item(s)')
                for i, item in enumerate(elem.value):
                    lines.append(f'{pad}  [item {i}]')
                    _walk(item, indent + 2)
            else:
                val = PHI_MASK if hide_phi and elem.keyword in PHI_FIELDS else _format_value(elem.value)
                lines.append(f'{pad}{elem.tag} {elem.name:<40} {elem.VR}: {val}')

    if hasattr(ds, 'file_meta') and ds.file_meta:
        lines.append('-- File meta information --')
        _walk(ds.file_meta, 0)
        lines.append('-- Dataset --')
    _walk(ds, 0)
    return lines


def print_dicom_info(
    dicom_dir: Union[str, os.PathLike],
    full: bool = False,
    tags: Optional[Sequence[str]] = None,
    recursive: bool = True,
    hide_phi: bool = False,
    file: Optional[TextIO] = None,
) -> Dict[str, List[Tuple[Path, Any]]]:
    """Print the DICOM header information for every series in a directory.

    Args:
        dicom_dir: Directory containing DICOM files (or a single DICOM file).
        full: Also dump the full header of the first file in each series.
        tags: Optional list of extra tags to print (keywords like ``'PatientWeight'``
            or hex like ``'0x00281053'`` / ``'0028,1053'``).
        recursive: Whether to descend into subdirectories.
        hide_phi: Mask patient-identifying fields.
        file: Output stream (defaults to ``sys.stdout``).

    Returns:
        The series dictionary from :func:`scan_dicom_dir`.

    """
    out = file or sys.stdout
    series = scan_dicom_dir(dicom_dir, recursive=recursive)
    if not series:
        print(f'No DICOM files found in {dicom_dir}', file=out)
        return series

    n_files = sum(len(v) for v in series.values())
    print(f'Found {n_files} DICOM file(s) in {len(series)} series under {dicom_dir}', file=out)

    for idx, (_uid, datasets) in enumerate(series.items(), 1):
        ds = datasets[0][1]
        title = f'Series {idx}/{len(series)}: {getattr(ds, "Modality", "?")} - {getattr(ds, "SeriesDescription", "")}'
        print('\n' + '=' * 80, file=out)
        print(title, file=out)
        print('=' * 80, file=out)

        for section, entries in summarize_series(datasets, hide_phi=hide_phi).items():
            print(f'[{section}]', file=out)
            for k, v in entries.items():
                print(f'  {k:<30} {v}', file=out)

        if tags:
            print('[Requested tags]', file=out)
            for t in tags:
                name, val = _get_element_value(ds, t)
                if val is not None and hide_phi and name.split()[-1] in PHI_FIELDS:
                    val = PHI_MASK
                print(f'  {name:<30} {_format_value(val) if val is not None else "<not present>"}', file=out)

        if full:
            print(f'[Full header: {datasets[0][0]}]', file=out)
            for line in full_header_lines(ds, hide_phi=hide_phi):
                print('  ' + line, file=out)
    return series


def main(argv: Optional[Sequence[str]] = None) -> None:
    """Command-line entry point; parses arguments and calls :func:`print_dicom_info`.

    Args:
        argv: Argument list (defaults to ``sys.argv[1:]``). Supports ``dicom_dir``, ``--full``, ``--tags``,
            ``--no-recursive``, ``--hide-phi`` and ``-o/--output`` (write to a text file instead of stdout).

    """
    parser = argparse.ArgumentParser(description='Print DICOM header information for each series in a directory.')
    parser.add_argument('dicom_dir', help='Directory containing DICOM files (or a single DICOM file).')
    parser.add_argument(
        '--full', action='store_true', help='Also dump the full header of the first file in each series.'
    )
    parser.add_argument(
        '--tags',
        nargs='+',
        default=None,
        help="Extra tags to print, as keywords or hex (e.g. PatientWeight 0x00281053 '0028,1053').",
    )
    parser.add_argument('--no-recursive', action='store_true', help='Do not descend into subdirectories.')
    parser.add_argument('--hide-phi', action='store_true', help='Mask patient-identifying fields.')
    parser.add_argument('-o', '--output', default=None, help='Write output to this text file instead of stdout.')
    args = parser.parse_args(argv)

    if args.output:
        os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
        with open(args.output, 'w') as f:
            print_dicom_info(
                args.dicom_dir,
                full=args.full,
                tags=args.tags,
                recursive=not args.no_recursive,
                hide_phi=args.hide_phi,
                file=f,
            )
    else:
        print_dicom_info(
            args.dicom_dir, full=args.full, tags=args.tags, recursive=not args.no_recursive, hide_phi=args.hide_phi
        )


if __name__ == '__main__':
    main()
