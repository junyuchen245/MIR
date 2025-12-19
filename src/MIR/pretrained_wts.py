import numpy as np

ModelWeights = {
    'VFA-LUMIR24-MonoModal': {'wts':'17XEfRYJbnrtCVhaBCOvQVOLkWhix9PAK', 'wts_key':'model_state_dict'},
    'VFA-LUMIR25-MultiModal': {'wts':'1cDY3isltI-uSCiivgP2zcx_5LeR8vIJ6', 'wts_key':'state_dict'},
    'TransMorphTVF-LUMIR24-MonoModal': {'wts':'1SSqI88l1MdrPJgE4Rn8pqXnVfZNPxtry', 'wts_key':'state_dict'},
    'VFA-SegHead': {'wts':'1gbWE5t_QwntpiY6vYEvf2wFi1YVm49Kg', 'wts_key':'state_dict'},
    'VFA-SynthHead': {'wts':'17Dn5xXTnVs0LB35IWOM22F6d5BORbgRH', 'wts_key':'state_dict'},
    'MedIA-TM-SPR-Beta-autoPET': {'wts':'1NCdoK4khv4j8JAjlgeJo6EB4dSQr83r6', 'wts_key':'state_dict'},
    'MedIA-TM-SPR-Gaussian-autoPET': {'wts':'1WqiR5YB8ypx-NUvYU_kQMb3edRVOLi09', 'wts_key':'state_dict'},
}

DatasetJSONs = {
    'LUMIR24': '1b0hyH7ggjCysJG-VGvo38XVE8bFVRMxb',
    'LUMIR25': '164Flc1C6oufONGimvpKlrNtq5t3obXEo'
}

FDG_atlas_autoPET = {
    'fdg_autoPET_CT_atlas_norm01_2.8x2.8x3.8': '1VGXnimeyN7bCzBOK4JKUFCYrOWin-IX6',
    'fdg_autoPET_SUV_atlas_norm01_2.8x2.8x3.8': '1aTZV9qv97Z9HswvkHWwISO9JsifzeeOh',
    'fdg_autoPET_organ_prob_atlas_105lbl_2.8x2.8x3.8': '1LZnT2DLuyNzZzzHk7u186CAk3XuUmedC',
    'fdg_autoPET_organ_seg_atlas_29lbl_2.8x2.8x3.8': '1spq_U5LmETRoyA-DmDtQ87eq5ZzBBQMF',
    'fdg_autoPET_CT_atlas_norm01_male_2.8x2.8x3.8': '10T01zx_LPPhLN2WoFCqtaZVkCDxen7sD',
    'fdg_autoPET_CT_atlas_norm01_female_2.8x2.8x3.8': '1rneTQoxq8n6IsM4iwg9yhhtZ_nSiyeVi',
    'fdg_autoPET_SUV_atlas_norm01_male_2.8x2.8x3.8': '1Gtb51rgVkpx4zZoESPB3Mozt59uhTSVs',
    'fdg_autoPET_SUV_atlas_norm01_female_2.8x2.8x3.8': '16XJnSTz5P1teLpa9xlyckvdm25n0wd_W',
}
