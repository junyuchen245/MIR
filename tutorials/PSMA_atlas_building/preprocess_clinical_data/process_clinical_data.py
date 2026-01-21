import pandas as pd
import re  # add regex for ID parsing
import json
import os
# Specify the path to your XLSX file
file_path = "/scratch2/jchen/DATA/PSMA_JHU/Preprocessed/follow-up_assignment8_new.xlsx"
output_dir = "/scratch2/jchen/DATA/PSMA_JHU/Preprocessed/clinical_jsons/"
os.makedirs(output_dir, exist_ok=True)
# Read the Excel file into a DataFrame using first row as headers and first column as index
df = pd.read_excel(file_path, header=0, index_col=2)

keys = ['Rad1 RECIP score (CR:1, PR:2, SD:3, PD:4)', 'Rad1 New leson (Yes:1, No:0)', 'Rad1 The location of new lesion (If have new lesion)', 'Rad2 RECIP score (CR:1, PR:2, SD:3, PD:4)', 'Rad2 New leson (Yes:1, No:0)', 
        'Rad2 The location of new lesion (If have new lesion)',  'Age', 'Race(white=1, 2=Black or African American，3 =Asian，4=other)', 'height ', 'weight (LBS)',	'PrePSMA Local(prostatectomy, radiation, brachytherapy，1=yes, 2=No）', 
        'Pre PSMA Focal(HIFU, Cryoablation),1=yes, 2=No', 'Pre PSMA Systemic Androgen Targeted(ADT, abiraterone, enzalutamide)1=yes, 2=No',	'Pre PSMA Systemic and cytotxic(taxanes ( docetaxel), pluvicto ( Lu))1=yes, 2=No', 
        'Initial PSA  (ng/ml)', 'PRE PSMA PSA  (ng/ml)', 'indications for PSMA-PET(1=Primary staging  2=Recurrence      -2a Biochemical      -2b Other recurrence,  3=Metastatic      -3a Oligometastatic       -3b Widespread metastatic, 4=Therapy Response,5=Research,6=Other', 
        'Gleason score ', 'T stage', 'Post PSMA Local(prostatectomy, radiation, brachytherapy，1=yes, 2=No）', 'Post PSMA Focal(HIFU, Cryoablation),1=yes, 2=No', 'Post PSMA Systemic Androgen Targeted(ADT, abiraterone, enzalutamide)1=yes, 2=No', 
        'Post PSMA Systemic and cytotxic(taxanes ( docetaxel), pluvicto ( Lu))1=yes, 2=No', 'Post PSMA PSA (ng/ml)', 'psa time', 'relapse(1=yes, no=2)', 'relapsetime', 'survival(1=Yes, No=2)', 'Date for surviival']

keys_cleaned = {
    'R1 RECIP score': 'Rad1 RECIP score (CR:1, PR:2, SD:3, PD:4)',
    'R1 New leson': 'Rad1 New leson (Yes:1, No:0)',
    'R1 Location of new lesion': 'Rad1 The location of new lesion (If have new lesion)',
    'R2 RECIP score': 'Rad2 RECIP score (CR:1, PR:2, SD:3, PD:4)',
    'R2 New leson': 'Rad2 New leson (Yes:1, No:0)',
    'R2 Location of new lesion': 'Rad2 The location of new lesion (If have new lesion)',
    'Age': 'Age',
    'Race': 'white=1, 2=Black or African American, 3 =Asian, 4=other',
    'height': 'height (m)',
    'weight': 'weight (lbs)',
    'Pre PSMA Local': 'prostatectomy, radiation, brachytherapy (1=yes, 2=No)',
    'Pre PSMA Focal': 'HIFU, Cryoablation (1=yes, 2=No)',
    'Pre PSMA Systemic Androgen Targeted': 'ADT, abiraterone, enzalutamide (1=yes, 2=No)',
    'Pre PSMA Systemic and cytotxic': 'taxanes (docetaxel), pluvicto (Lu) (1=yes, 2=No)',
    'Initial PSA': 'Initial PSA (ng/ml)',
    'PRE PSMA PSA': 'PRE PSMA PSA (ng/ml)',
    'Indications for PSMA-PET': '1=Primary staging, 2=Recurrence -2a Biochemical -2b Other recurrence, 3=Metastatic -3a Oligometastatic -3b Widespread metastatic, 4=Therapy Response, 5=Research, 6=Other',
    'Gleason score': 'Gleason score',
    'T stage': 'T stage',
    'Post PSMA Local': 'prostatectomy, radiation, brachytherapy (1=yes, 2=No)',
    'Post PSMA Focal': 'HIFU, Cryoablation (1=yes, 2=No)',
    'Post PSMA Systemic Androgen Targeted': 'ADT, abiraterone, enzalutamide (1=yes, 2=No)',
    'Post PSMA Systemic and cytotxic': 'taxanes (docetaxel), pluvicto (Lu) (1=yes, 2=No)',
    'Post PSMA PSA': 'Post PSMA PSA (ng/ml)',
    'psa time': 'psa time',
    'relapse': '1=yes, no=2',
    'relapsetime': 'relapsetime',
    'survival': '1=Yes, No=2',
    'Date for survival': 'Date for survival'
}
pd.set_option('display.max_colwidth', None)
pd.set_option('display.max_columns', None)
print(len(keys), len(keys_cleaned))
#print(df)
# Use the keys to select corresponding columns (warn and keep available ones if some are missing)
try:
    df_selected = df.loc[:, keys]
except KeyError:
    present = [k for k in keys if k in df.columns]
    missing = [k for k in keys if k not in df.columns]
    print(f"Warning: missing columns not found in DataFrame: {missing}")
    df_selected = df.loc[:, present]

# Insert hyphens into 8-digit dates within subject IDs, e.g., 20220629 -> 2022-06-29
def insert_date_hyphens(sid: str) -> str:
    parts = str(sid).split('_')
    if len(parts) >= 2 and re.fullmatch(r'\d{8}', parts[1]):
        date = parts[1]
        formatted = f"{date[:4]}-{date[4:6]}-{date[6:8]}"
        return '_'.join([parts[0], formatted] + parts[2:])
    return sid

df_selected = df_selected.copy()
df_selected.index = [insert_date_hyphens(idx) for idx in df_selected.index]

# Map original Excel column names to cleaned keys used for description lookup
key_name_map = {
    'Rad1 RECIP score (CR:1, PR:2, SD:3, PD:4)': 'R1 RECIP score',
    'Rad1 New leson (Yes:1, No:0)': 'R1 New lesion',
    'Rad1 The location of new lesion (If have new lesion)': 'R1 Location of new lesion',
    'Rad2 RECIP score (CR:1, PR:2, SD:3, PD:4)': 'R2 RECIP score',
    'Rad2 New leson (Yes:1, No:0)': 'R2 New lesion',
    'Rad2 The location of new lesion (If have new lesion)': 'R2 Location of new lesion',
    'Age': 'Age',
    'Race(white=1, 2=Black or African American，3 =Asian，4=other)': 'Race',
    'height ': 'height',
    'weight (LBS)': 'weight',
    'PrePSMA Local(prostatectomy, radiation, brachytherapy，1=yes, 2=No）': 'Pre PSMA Local',
    'Pre PSMA Focal(HIFU, Cryoablation),1=yes, 2=No': 'Pre PSMA Focal',
    'Pre PSMA Systemic Androgen Targeted(ADT, abiraterone, enzalutamide)1=yes, 2=No': 'Pre PSMA Systemic Androgen Targeted',
    'Pre PSMA Systemic and cytotxic(taxanes ( docetaxel), pluvicto ( Lu))1=yes, 2=No': 'Pre PSMA Systemic and cytotxic',
    'Initial PSA  (ng/ml)': 'Initial PSA',
    'PRE PSMA PSA  (ng/ml)': 'PRE PSMA PSA',
    'indications for PSMA-PET(1=Primary staging  2=Recurrence      -2a Biochemical      -2b Other recurrence,  3=Metastatic      -3a Oligometastatic       -3b Widespread metastatic, 4=Therapy Response,5=Research,6=Other': 'Indications for PSMA-PET',
    'Gleason score ': 'Gleason score',
    'T stage': 'T stage',
    'Post PSMA Local(prostatectomy, radiation, brachytherapy，1=yes, 2=No）': 'Post PSMA Local',
    'Post PSMA Focal(HIFU, Cryoablation),1=yes, 2=No': 'Post PSMA Focal',
    'Post PSMA Systemic Androgen Targeted(ADT, abiraterone, enzalutamide)1=yes, 2=No': 'Post PSMA Systemic Androgen Targeted',
    'Post PSMA Systemic and cytotxic(taxanes ( docetaxel), pluvicto ( Lu))1=yes, 2=No': 'Post PSMA Systemic and cytotxic',
    'Post PSMA PSA (ng/ml)': 'Post PSMA PSA',
    'psa time': 'psa time',
    'relapse(1=yes, no=2)': 'relapse',
    'relapsetime': 'relapsetime',
    'survival(1=Yes, No=2)': 'survival',
    'Date for surviival': 'Date for survival',
}

# Convert all entries to strings; NaN -> ''
df_str = df_selected.applymap(lambda x: '' if pd.isna(x) else str(x).strip())

# Helper to robustly extract a single string value (handles duplicates returning Series)
def get_cell_as_str(df_like: pd.DataFrame, idx, col) -> str:
    v = df_like.loc[idx, col]
    if isinstance(v, pd.Series):
        vals = [str(x).strip() for x in v.tolist() if str(x).strip() != '']
        return ' | '.join(vals) if vals else ''
    return str(v).strip()

# Only iterate over columns that are present
keys_present = [k for k in keys if k in df_str.columns]

for subject_id in df_str.index:
    if 'PSMA' in subject_id:
        continue
    data_json = {'subject id': subject_id}
    for orig_key in keys_present:
        cleaned_key = key_name_map.get(orig_key, orig_key)
        description = keys_cleaned.get(cleaned_key, '')
        value_str = get_cell_as_str(df_str, subject_id, orig_key)
        if " | " in value_str:
            value_str = value_str.split(" | ")[0]
        data_json[cleaned_key] = {
            'description': description,
            'data': value_str,
        }
    json_filename = f"{subject_id}.json"
    with open(os.path.join(output_dir, json_filename), 'w', encoding='utf-8') as json_file:
        json.dump(data_json, json_file, indent=4, ensure_ascii=False)