"""Reference label tables for MIR datasets."""

import numpy as np

totalseg_v1_105labels = {
    0: "Background",
    1: "Spleen",
    2: "Right kidney",
    3: "Left kidney",
    4: "Gallbladder",
    5: "Liver",
    6: "Stomach",
    7: "Aorta",
    8: "Inferior vena cava",
    9: "Portal vein and splenic vein (hepatic portal vein)",
    10: "Pancreas",
    11: "Right adrenal gland (suprarenal gland)",
    12: "Left adrenal gland (suprarenal gland)",
    13: "Left lung upper lobe (superior lobe)",
    14: "Left lung lower lobe (inferior lobe)",
    15: "Right lung upper lobe (superior lobe)",
    16: "Right lung middle lobe",
    17: "Right lung lower lobe (inferior lobe)",
    18: "Vertebra L5",
    19: "Vertebra L4",
    20: "Vertebra L3",
    21: "Vertebra L2",
    22: "Vertebra L1",
    23: "Vertebra T12",
    24: "Vertebra T11",
    25: "Vertebra T10",
    26: "Vertebra T9",
    27: "Vertebra T8",
    28: "Vertebra T7",
    29: "Vertebra T6",
    30: "Vertebra T5",
    31: "Vertebra T4",
    32: "Vertebra T3",
    33: "Vertebra T2",
    34: "Vertebra T1",
    35: "Vertebra C7",
    36: "Vertebra C6",
    37: "Vertebra C5",
    38: "Vertebra C4",
    39: "Vertebra C3",
    40: "Vertebra C2",
    41: "Vertebra C1",
    42: "Esophagus",
    43: "Trachea",
    44: "Heart myocardium",
    45: "Left atrium",
    46: "Left ventricle",
    47: "Right atrium",
    48: "Right ventricle",
    49: "Pulmonary artery",
    50: "Brain",
    51: "Left common iliac artery",
    52: "Right common iliac artery",
    53: "Left common iliac vein",
    54: "Right common iliac vein",
    55: "Small bowel (small intestine)",
    56: "Duodenum",
    57: "Colon",
    58: "Left rib 1",
    59: "Left rib 2",
    60: "Left rib 3",
    61: "Left rib 4",
    62: "Left rib 5",
    63: "Left rib 6",
    64: "Left rib 7",
    65: "Left rib 8",
    66: "Left rib 9",
    67: "Left rib 10",
    68: "Left rib 11",
    69: "Left rib 12",
    70: "Right rib 1",
    71: "Right rib 2",
    72: "Right rib 3",
    73: "Right rib 4",
    74: "Right rib 5",
    75: "Right rib 6",
    76: "Right rib 7",
    77: "Right rib 8",
    78: "Right rib 9",
    79: "Right rib 10",
    80: "Right rib 11",
    81: "Right rib 12",
    82: "Left humerus",
    83: "Right humerus",
    84: "Left scapula",
    85: "Right scapula",
    86: "Left clavicle",
    87: "Right clavicle",
    88: "Left femur",
    89: "Right femur",
    90: "Left hip bone",
    91: "Right hip bone",
    92: "Sacrum",
    93: "Face",
    94: "Left gluteus maximus",
    95: "Right gluteus maximus",
    96: "Left gluteus medius",
    97: "Right gluteus medius",
    98: "Left gluteus minimus",
    99: "Right gluteus minimus",
    100: "Left autochthonous muscle",
    101: "Right autochthonous muscle",
    102: "Left iliopsoas muscle",
    103: "Right iliopsoas muscle",
    104: "Urinary bladder",
}

totalseg_v2_118labels = {
    0: "Background",
    1: 'Spleen',
    2: 'Right kidney',
    3: 'Left kidney',
    4: 'Gallbladder',
    5: 'Liver',
    6: 'Stomach',
    7: 'Pancreas',
    8: 'Right adrenal gland (suprarenal gland)',
    9: 'Left adrenal gland (suprarenal gland)',
    10: 'Left lung upper lobe (superior lobe)',
    11: 'Left lung lower lobe (inferior lobe)',
    12: 'Right lung upper lobe (superior lobe)',
    13: 'Right lung middle lobe',
    14: 'Right lung lower lobe (inferior lobe)',
    15: 'Esophagus',
    16: 'Trachea',
    17: 'Thyroid gland',
    18: 'Small bowel (small intestine)',
    19: 'Duodenum',
    20: 'Colon',
    21: 'Urinary bladder',
    22: 'Prostate',
    23: 'Left kidney cyst',
    24: 'Right kidney cyst',
    25: 'Sacrum',
    26: 'Vertebra S1',
    27: 'Vertebra L5',
    28: 'Vertebra L4',
    29: 'Vertebra L3',
    30: 'Vertebra L2',
    31: 'Vertebra L1',
    32: 'Vertebra T12',
    33: 'Vertebra T11',
    34: 'Vertebra T10',
    35: 'Vertebra T9',
    36: 'Vertebra T8',
    37: 'Vertebra T7',
    38: 'Vertebra T6',
    39: 'Vertebra T5',
    40: 'Vertebra T4',
    41: 'Vertebra T3',
    42: 'Vertebra T2',
    43: 'Vertebra T1',
    44: 'Vertebra C7',
    45: 'Vertebra C6',
    46: 'Vertebra C5',
    47: 'Vertebra C4',
    48: 'Vertebra C3',
    49: 'Vertebra C2',
    50: 'Vertebra C1',
    51: 'Heart',
    52: 'Aorta',
    53: 'Pulmonary vein',
    54: 'Brachiocephalic trunk',
    55: 'Right subclavian artery',
    56: 'Left subclavian artery',
    57: 'Right common carotid artery',
    58: 'Left common carotid artery',
    59: 'Left brachiocephalic vein',
    60: 'Right brachiocephalic vein',
    61: 'Left atrial appendage',
    62: 'Superior vena cava',
    63: 'Inferior vena cava',
    64: 'Portal vein and splenic vein (hepatic portal vein)',
    65: 'Left iliac artery (common iliac artery)',
    66: 'Right iliac artery (common iliac artery)',
    67: 'Left iliac vein (common iliac vein)',
    68: 'Right iliac vein (common iliac vein)',
    69: 'Left humerus',
    70: 'Right humerus',
    71: 'Left scapula',
    72: 'Right scapula',
    73: 'Left clavicle',
    74: 'Right clavicle',
    75: 'Left femur',
    76: 'Right femur',
    77: 'Left hip bone',
    78: 'Right hip bone',
    79: 'Spinal cord',
    80: 'Left gluteus maximus muscle',
    81: 'Right gluteus maximus muscle',
    82: 'Left gluteus medius muscle',
    83: 'Right gluteus medius muscle',
    84: 'Left gluteus minimus muscle',
    85: 'Right gluteus minimus muscle',
    86: 'Left autochthonous muscle',
    87: 'Right autochthonous muscle',
    88: 'Left iliopsoas muscle',
    89: 'Right iliopsoas muscle',
    90: 'Brain',
    91: 'Skull',
    92: 'Left rib 1',
    93: 'Left rib 2',
    94: 'Left rib 3',
    95: 'Left rib 4',
    96: 'Left rib 5',
    97: 'Left rib 6',
    98: 'Left rib 7',
    99: 'Left rib 8',
    100: 'Left rib 9',
    101: 'Left rib 10',
    102: 'Left rib 11',
    103: 'Left rib 12',
    104: 'Right rib 1',
    105: 'Right rib 2',
    106: 'Right rib 3',
    107: 'Right rib 4',
    108: 'Right rib 5',
    109: 'Right rib 6',
    110: 'Right rib 7',
    111: 'Right rib 8',
    112: 'Right rib 9',
    113: 'Right rib 10',
    114: 'Right rib 11',
    115: 'Right rib 12',
    116: 'Sternum',
    117: 'Costal cartilages'
}

merged_regions_coarse_from_totalsegv2 = {
    "Background": [0],

    # Brain and head
    "Brain": [90],
    "Skull": [91],

    # Thoracic organs
    "Lungs": [10, 11, 12, 13, 14],
    "Heart_Mediastinum_Vessels": [
        51, 52, 53, 54, 55, 56, 57, 58,
        59, 60, 61, 62, 63, 64
    ],

    # Abdominal solid organs
    "Liver": [5],
    "Spleen": [1],
    "Kidneys_and_Cysts": [2, 3, 23, 24],
    "Adrenals": [8, 9],
    "Pancreas": [7],
    "Gallbladder": [4],

    # Gastrointestinal tract
    "Stomach": [6],
    "Small_bowel": [18, 19],
    "Colon": [20],

    # Pelvic organs
    "Urinary_bladder": [21],
    "Prostate_region": [22],

    # Axial skeleton
    "Cervical_spine": [44, 45, 46, 47, 48, 49, 50],
    "Thoracic_spine": [32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43],
    "Lumbar_spine": [27, 28, 29, 30, 31],
    "Sacrum_S1": [25, 26],

    # Ribs and chest wall bones
    "Ribs_and_Sternum": [
        92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103,
        104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115,
        116  # sternum
    ],
    "Clavicles_and_Scapulae": [71, 72, 73, 74],

    # Appendicular skeleton
    "Humeri": [69, 70],
    "Femora": [75, 76],
    "Hip_bones": [77, 78],

    # Major muscle groups (optional, can be one big region)
    "Gluteal_muscles": [80, 81, 82, 83, 84, 85],
    "Paraspinal_muscles": [86, 87],
    "Iliopsoas_muscles": [88, 89],

    # Spinal cord (rarely relevant for PSMA mets, but keep separate)
    "Spinal_cord": [79],

    # Costal_cartilage (rarely needed, can be merged with ribs if you like)
    "Costal_cartilages": [117],

    # Thyroid, trachea, esophagus (optional separate soft tissue region)
    "Neck_and_esophagus_soft_tissue": [15, 16, 17]
}

totalseg_merged_40labels = {
    0: "Background",
    1: "Spleen",
    2: "Right kidney",
    3: "Left kidney",
    4: "Gallbladder",
    5: "Liver",
    6: "Stomach",
    7: "Aorta",
    8: "Inferior vena cava",
    9: "Portal vein and splenic vein",
    10: "Pancreas",
    11: "Right adrenal gland",
    12: "Left adrenal gland",
    13: "Left lung (upper and lower lobes)",
    14: "Right lung (upper, middle, and lower lobes)",
    15: "Vertebrae (C1 to L5)",
    16: "Esophagus",
    17: "Trachea",
    18: "Heart myocardium",
    19: "Left atrium",
    20: "Left ventricle",
    21: "Right atrium",
    22: "Right ventricle",
    23: "Pulmonary artery",
    24: "Small bowel",
    25: "Duodenum",
    26: "Colon",
    27: "Ribs (left and right, 1–12)",
    28: "Left upper limb bones (humerus, scapula, clavicle)",
    29: "Right upper limb bones (humerus, scapula, clavicle)",
    30: "Left lower limb bones (femur, hip bone)",
    31: "Right lower limb bones (femur, hip bone)",
    32: "Sacrum",
    33: "Left gluteal muscles (maximus, medius, minimus)",
    34: "Right gluteal muscles (maximus, medius, minimus)",
    35: "Left autochthonous muscle",
    36: "Right autochthonous muscle",
    37: "Left iliopsoas muscle",
    38: "Right iliopsoas muscle",
    39: "Urinary bladder",
}

totalseg_merged_29labels = {
    0: "Background",
    1: "Spleen",
    2: "Kidneys (right and left)",
    3: "Gallbladder",
    4: "Liver",
    5: "Stomach",
    6: "Aorta",
    7: "Inferior vena cava",
    8: "Portal vein and splenic vein",
    9: "Pancreas",
    10: "Right adrenal gland",
    11: "Left adrenal gland",
    12: "Lungs (left and right lobes)",
    13: "Vertebrae (C1 to L5)",
    14: "Esophagus",
    15: "Trachea",
    16: "Heart myocardium",
    17: "Left atrium",
    18: "Left ventricle",
    19: "Right atrium",
    20: "Right ventricle",
    21: "Pulmonary artery",
    22: "Intestine (small bowel, duodenum, colon)",
    23: "Ribs (left and right, 1–12)",
    24: "Upper limb bones (left side)",
    25: "Upper limb bones (right side)",
    26: "Lower limb bones (left side)",
    27: "Lower limb bones (right side)",
    28: "Urinary bladder",
}

def remap_totalsegmentator_lbls(lbl, total_seg_version='v1', label_scheme='40lbls'):
    groups29_from_40 = [
        [1],                #  1  Spleen
        [2, 3],             #  2  Kidneys (R+L)
        [4],                #  3  Gallbladder
        [5],                #  4  Liver
        [6],                #  5  Stomach
        [7],                #  6  Aorta
        [8],                #  7  Inferior vena cava
        [9],                #  8  Portal & splenic veins
        [10],               #  9  Pancreas
        [11],               # 10  Adrenal right
        [12],               # 11  Adrenal left
        [13, 14],           # 12  Lungs (left+right)
        [15],               # 13  Vertebrae
        [16],               # 14  Esophagus
        [17],               # 15  Trachea
        [18],               # 16  Heart myocardium
        [19],               # 17  Left atrium
        [20],               # 18  Left ventricle
        [21],               # 19  Right atrium
        [22],               # 20  Right ventricle
        [23],               # 21  Pulmonary artery
        [24, 25, 26],       # 22  Intestine (small bowel + duodenum + colon)
        [27],               # 23  Ribs
        [28, 29, 30, 31, 32], # 24 Upper/lower limb bones (L/R) + sacrum
        [33, 34],           # 25  Gluteal muscles (L+R)
        [35, 36],           # 26  Autochthonous muscles (L+R)
        [37, 38],           # 27  Iliopsoas muscles (L+R)
        [39],               # 28  Urinary bladder
    ]
    
    grouping_table_v1_40 = [
        [1],                          # 1  Spleen
        [2],                          # 2  Kidney right
        [3],                          # 3  Kidney left
        [4],                          # 4  Gallbladder
        [5],                          # 5  Liver
        [6],                          # 6  Stomach
        [7],                          # 7  Aorta
        [8],                          # 8  Inferior vena cava
        [9],                          # 9  Portal & splenic veins
        [10],                         # 10 Pancreas
        [11],                         # 11 Adrenal right
        [12],                         # 12 Adrenal left
        [13, 14],                     # 13 Left lung (upper+lower)
        [15, 16, 17],                 # 14 Right lung (upper+middle+lower)
        list(range(18, 42)),          # 15 Vertebrae C1..L5 (18..41)
        [42],                         # 16 Esophagus
        [43],                         # 17 Trachea
        [44],                         # 18 Heart myocardium
        [45],                         # 19 Left atrium
        [46],                         # 20 Left ventricle
        [47],                         # 21 Right atrium
        [48],                         # 22 Right ventricle
        [49],                         # 23 Pulmonary artery
        [55],                         # 24 Small bowel
        [56],                         # 25 Duodenum
        [57],                         # 26 Colon
        list(range(58, 82)),          # 27 Ribs L1..L12 and R1..R12 (58..81)
        [82, 84, 86],                 # 28 Upper limb bones (left) humerus+scapula+clavicle
        [83, 85, 87],                 # 29 Upper limb bones (right)
        [88, 90],                     # 30 Lower limb bones (left) femur+hip bone
        [89, 91],                     # 31 Lower limb bones (right)
        [92],                         # 32 Sacrum
        [94, 96, 98],                 # 33 Gluteal muscles (left)
        [95, 97, 99],                 # 34 Gluteal muscles (right)
        [100],                        # 35 Autochthonous (left)
        [101],                        # 36 Autochthonous (right)
        [102],                        # 37 Iliopsoas (left)
        [103],                        # 38 Iliopsoas (right)
        [104],                        # 39 Urinary bladder
    ]
    
    grouping_table_v2_40 = [
        [1],                           # 1 spleen
        [2],                           # 2 kidney_right
        [3],                           # 3 kidney_left
        [4],                           # 4 gallbladder
        [5],                           # 5 liver
        [6],                           # 6 stomach
        [52],                          # 7 aorta
        [63],                          # 8 inferior_vena_cava
        [64],                          # 9 portal_vein_and_splenic_vein
        [7],                           # 10 pancreas
        [8],                           # 11 adrenal_gland_right
        [9],                           # 12 adrenal_gland_left
        [10, 11],                      # 13 lung upper+lower lobe left
        [12, 13, 14],                  # 14 lung upper/middle/lower right
        [27, 28, 29, 30, 31, 32, 33,   # 15 vertebrae L5 to C1
         34, 35, 36, 37, 38, 39, 40,
         41, 42, 43, 44, 45, 46, 47,
         48, 49, 50, 26, 25],          # Include S1 and sacrum (old grouping had sacrum separately, but L+T+C+S grouped)
        [15],                          # 16 esophagus
        [16],                          # 17 trachea
        [51],                          # 18 heart (old version separated myocardium/atrium/ventricle but grouped into one label)
        [61],                          # 19 heart_atrium_left (moved to same group as myocardium in old grouping)
        [46],                          # 20 heart_ventricle_left -> (merged)
        [47],                          # 21 heart_atrium_right -> (merged)
        [48],                          # 22 heart_ventricle_right -> (merged)
        [53],                          # 23 pulmonary vein (old had pulmonary artery, but analogous major pulmonary vessel group)
        [18],                          # 24 small_bowel
        [19],                          # 25 duodenum
        [20],                          # 26 colon
        [92, 93, 94, 95, 96, 97, 98,   # 27 left ribs 1-12
         99, 100, 101, 102, 103,
         104, 105, 106, 107, 108,      # right ribs 1-12
         109, 110, 111, 112, 113,
         114, 115],
        [69, 71, 73],                  # 28 humerus_left, scapula_left, clavicula_left
        [70, 72, 74],                  # 29 humerus_right, scapula_right, clavicula_right
        [75, 77],                      # 30 femur_left, hip_left
        [76, 78],                      # 31 femur_right, hip_right
        [25],                          # 32 sacrum (old had sacrum alone too, but here also part of vertebrae group above if desired)
        [80, 82, 84],                  # 33 gluteus_maximus_left, gluteus_medius_left, gluteus_minimus_left
        [81, 83, 85],                  # 34 gluteus_maximus_right, gluteus_medius_right, gluteus_minimus_right
        [86],                          # 35 autochthon_left
        [87],                          # 36 autochthon_right
        [88],                          # 37 iliopsoas_left
        [89],                          # 38 iliopsoas_right
        [21]                           # 39 urinary_bladder
    ]
    
    grouping_table_biological_meaningful_fromv2_118 = [
        #[0],                                            # 0 "Background": 0
         
        # Brain and head
        #[90, 91],                                           # 1 "Brain": 
        #[91],                                           # 2 "Skull": 

        # Thoracic organs
        [10, 11, 12, 13, 14],                           # 3 "Lungs": 1
        [51, 52, 53, 54, 55, 56, 57, 58,
        59, 60, 61, 62, 63, 64],                        # 4 "Heart_Mediastinum_Vessels": 2 

        # Abdominal solid organs
        #[5],                                            # 5 "Liver": 
        #[1],                                            # 6 "Spleen": 
        #[2, 3, 23, 24],                                 # 7 "Kidneys_and_Cysts": 
        #[8, 9],                                         # 8 "Adrenals": 
        [7],                                            # 9 "Pancreas": 3
        [4],                                            # 10 "Gallbladder": 4 

        # Gastrointestinal tract
        [6],                                           # 11 "Stomach": 5
        #[18, 19],                                      # 12 "Small_bowel": 
        #[20],                                          # 13 "Colon": 

        # Pelvic organs
        #[21, 22],                                          # 14 "Urinary_bladder": 
        #[22],                                          # 15 "Prostate_region": 

        # Axial skeleton
        [44, 45, 46, 47, 48, 49, 50],                     # 16 "Cervical_spine": 6 
        [32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43], # 17 "Thoracic_spine": 7
        [27, 28, 29, 30, 31],                             # 18 "Lumbar_spine":   8
        [25, 26],                                         # 19 "Sacrum_S1":      9

        # Ribs and chest wall bones
        [92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 105, 
         106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116], # 20 "Ribs_and_Sternum": 10 
        [71, 72, 73, 74],                                        # 21 "Clavicles_and_Scapulae": 11

        # Appendicular skeleton
        [69, 70, 75, 76, 77, 78],                                       # 22 "Humeri": 12
        #[75, 76],                                       # 23 "Femora":
        #[77, 78],                                       # 24 "Hip_bones":

        # Major muscle groups (optional, can be one big region)
        #[80, 81, 82, 83, 84, 85],                       # 25 "Gluteal_muscles":
        #[86, 87],                                       # 26 "Paraspinal_muscles":
        #[88, 89],                                       # 27 "Iliopsoas_muscles":

        # Spinal cord (rarely relevant for PSMA mets, but keep separate)
        [79],                                          # 28 "Spinal_cord": 13

        # Costal_cartilage (rarely needed, can be merged with ribs if you like)
        [117],                                         # 29 "Costal_cartilages": 14

        # Thyroid, trachea, esophagus (optional separate soft tissue region)
        [15, 16, 17]                                   # 30 "Neck_and_esophagus_soft_tissue": 15
    ]
    
    
    if total_seg_version == 'v1':
        grouping_table = grouping_table_v1_40
    elif total_seg_version == 'v2':
        grouping_table = grouping_table_v2_40
    elif total_seg_version == 'v2_biological_meaningful':
        grouping_table = grouping_table_biological_meaningful_fromv2_118
    else:
        raise ValueError(f"Unknown total_seg_version: {total_seg_version}")

    label_out = np.zeros_like(lbl)
    for idx, item in enumerate(grouping_table):
        for seg_i in item:
            label_out[lbl == seg_i] = idx + 1
    if label_scheme == '29lbls':
        lbl = label_out.copy()
        label_out = np.zeros_like(lbl)
        for idx, group in enumerate(groups29_from_40):
            for seg_i in group:
                label_out[lbl == seg_i] = idx + 1
        return label_out
    return label_out