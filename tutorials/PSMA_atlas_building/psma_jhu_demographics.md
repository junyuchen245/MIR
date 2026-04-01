| Variable | Value |
| --- | --- |
| Dataset (patients) | 130 |
| Age (y) (mean ± SD) | 72.55 ± 8.11 |
| Sex |  |
|   Men | 130 |
|   Women | 0 |
|   Unknown | 0 |
| Race |  |
|   White | 109 |
|   Black or African American | 16 |
|   Asian | 0 |
|   Other | 3 |
|   Unknown | 2 |
| Gleason score |  |
|   NA | 8 |
|   ≤6 | 5 |
|   7 | 41 |
|   8 | 23 |
|   9 | 44 |
|   10 | 9 |
| Initial PSA level (ng/mL) | 7.65 (1.20–864.00) |
| Pre-PSMA PSA level (ng/mL) | 3.20 (0.19–138.40) |
| Post-PSMA PSA level (ng/mL) | 0.60 (0.01–192.00) |
| PSA follow-up interval (mo) | 4.27 (0.00–89.43) |
| Tracer |  |
|   DCFPyL / Pylarify | 121 |
|   68Ga-PSMA | 1 |
|   Fluciclovine | 1 |
|   Solution / unspecified | 4 |
|   Missing | 3 |
| Radionuclide |  |
|   18F | 126 |
|   68Ga | 1 |
|   Missing | 3 |
| Injected dose (MBq) | 322.3 (144.7–362.2) |
| Scanner model |  |
|   Biograph128_mCT | 99 |
|   Discovery RX | 24 |
|   Missing | 3 |
|   Biograph128_mCT 4R | 2 |
|   Discovery IQ | 1 |
|   Biograph64_mCT | 1 |
| Post-PSMA therapy |  |
|   NA | 2 |
|   None | 4 |
|   Local | 21 |
|   Systemic androgen-targeted | 50 |
|   Systemic and cytotoxic | 53 |

**Notes**
- This summary is restricted to patients with available processed images in /scratch2/jchen/DATA/PSMA_JHU/Preprocessed/JHU; 130 of 201 patients had scans meeting the 'ct-suv' image filter.
- Sex is not stored in these JSONs; this summary infers an all-male cohort because the dataset is prostate cancer.
- PSA doubling time is not available as a direct field in these JSONs.
- The displayed PSA follow-up interval is the time from 'psa time' to 'relapsetime' when both dates exist; it is not a doubling-time estimate.
- Tracer, radionuclide, dose, and scanner-model statistics come from matched scan-level DICOM metadata JSONs in JHU_dicom_info.
- When explicit radionuclide coding is absent but the tracer label is a known agent such as DCFPyL/Pylarify, the radionuclide is inferred from the tracer identity.
- 11 selected scans used a unique same-date fallback match because no exact scan-name DICOM JSON was found.
- Some selected scans do not have matched radiopharmaceutical metadata in JHU_dicom_info, so their tracer statistics are reported as Missing.
- Dominant therapy mode collapses combination treatments into a single highest-intensity category.
- 122 clinical scan records were excluded because the required processed images were not present.
