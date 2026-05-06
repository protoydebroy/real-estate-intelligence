# ============================================================
# geo_coords.py  —  Lat/Lng lookup for Gurgaon + Bangalore
# ============================================================
# Built-in coordinate dictionaries for the major locations in
# both cities, with fuzzy matching to handle the messy sector
# strings in our dataset (e.g. "soukya road, whitefield, bangalore"
# matches "whitefield").
# ============================================================

import re

# ─────────────────────────────────────────────────────────────
# GURGAON SECTORS  →  approximate lat/lng centroids
# (data sourced from public OpenStreetMap data)
# ─────────────────────────────────────────────────────────────

GURGAON_COORDS = {
    'sector 1':       (28.4595, 77.0266),
    'sector 2':       (28.4795, 77.0426),
    'sector 3':       (28.4760, 77.0312),
    'sector 4':       (28.4691, 77.0289),
    'sector 5':       (28.4685, 77.0411),
    'sector 6':       (28.4612, 77.0400),
    'sector 7':       (28.4590, 77.0355),
    'sector 8':       (28.4536, 77.0321),
    'sector 9':       (28.4498, 77.0265),
    'sector 10':      (28.4462, 77.0227),
    'sector 11':      (28.4567, 77.0162),
    'sector 12':      (28.4631, 77.0212),
    'sector 13':      (28.4696, 77.0182),
    'sector 14':      (28.4749, 77.0317),
    'sector 15':      (28.4612, 77.0357),
    'sector 17':      (28.4655, 77.0489),
    'sector 21':      (28.4828, 77.0773),
    'sector 22':      (28.4843, 77.0825),
    'sector 23':      (28.4769, 77.0683),
    'sector 24':      (28.4723, 77.0612),
    'sector 25':      (28.4651, 77.0759),
    'sector 26':      (28.4622, 77.0863),
    'sector 27':      (28.4548, 77.0852),
    'sector 28':      (28.4708, 77.0825),
    'sector 30':      (28.4476, 77.0780),
    'sector 31':      (28.4475, 77.0660),
    'sector 33':      (28.4413, 77.0561),
    'sector 36':      (28.4385, 77.0480),
    'sector 37':      (28.4319, 77.0408),
    'sector 37d':     (28.4287, 77.0356),
    'sector 38':      (28.4247, 77.0317),
    'sector 39':      (28.4202, 77.0270),
    'sector 40':      (28.4156, 77.0223),
    'sector 41':      (28.4097, 77.0177),
    'sector 43':      (28.4378, 77.0934),
    'sector 45':      (28.4502, 77.0997),
    'sector 46':      (28.4597, 77.1077),
    'sector 47':      (28.4666, 77.1145),
    'sector 48':      (28.4239, 77.0617),
    'sector 49':      (28.4181, 77.0664),
    'sector 50':      (28.4129, 77.0712),
    'sector 51':      (28.4476, 77.0934),
    'sector 52':      (28.4413, 77.1004),
    'sector 53':      (28.4347, 77.1085),
    'sector 54':      (28.4286, 77.1166),
    'sector 55':      (28.4220, 77.1245),
    'sector 56':      (28.4170, 77.1343),
    'sector 57':      (28.4112, 77.1426),
    'sector 58':      (28.4029, 77.1505),
    'sector 59':      (28.3956, 77.1572),
    'sector 60':      (28.3886, 77.1640),
    'sector 61':      (28.3812, 77.1714),
    'sector 62':      (28.3735, 77.1786),
    'sector 63':      (28.3661, 77.1852),
    'sector 63a':     (28.3589, 77.1903),
    'sector 65':      (28.3924, 77.0654),
    'sector 66':      (28.3853, 77.0594),
    'sector 67':      (28.3784, 77.0532),
    'sector 67a':     (28.3717, 77.0479),
    'sector 68':      (28.3649, 77.0428),
    'sector 69':      (28.3570, 77.0392),
    'sector 70':      (28.3494, 77.0354),
    'sector 70a':     (28.3439, 77.0316),
    'sector 71':      (28.3380, 77.0277),
    'sector 72':      (28.3315, 77.0244),
    'sector 73':      (28.3253, 77.0210),
    'sector 74':      (28.3196, 77.0184),
    'sector 76':      (28.3866, 77.0204),
    'sector 77':      (28.3801, 77.0173),
    'sector 78':      (28.3737, 77.0142),
    'sector 79':      (28.3670, 77.0114),
    'sector 80':      (28.3603, 77.0085),
    'sector 81':      (28.3554, 77.0055),
    'sector 82':      (28.3495, 77.0026),
    'sector 82a':     (28.3447, 77.0008),
    'sector 83':      (28.3401, 76.9989),
    'sector 84':      (28.3341, 76.9962),
    'sector 85':      (28.3280, 76.9931),
    'sector 86':      (28.3221, 76.9897),
    'sector 88':      (28.3092, 76.9843),
    'sector 88a':     (28.3046, 76.9826),
    'sector 89':      (28.3008, 76.9795),
    'sector 90':      (28.2981, 76.9745),
    'sector 91':      (28.2935, 76.9719),
    'sector 92':      (28.2878, 76.9700),
    'sector 93':      (28.2832, 76.9678),
    'sector 95':      (28.2758, 76.9649),
    'sector 99':      (28.5051, 76.9683),
    'sector 102':     (28.5174, 76.9744),
    'sector 103':     (28.5232, 76.9779),
    'sector 104':     (28.5285, 76.9821),
    'sector 105':     (28.5339, 76.9870),
    'sector 106':     (28.5398, 76.9923),
    'sector 107':     (28.5460, 76.9977),
    'sector 108':     (28.5518, 77.0040),
    'sector 109':     (28.5577, 77.0098),
    'sector 110':     (28.5641, 77.0156),
    'sector 111':     (28.5702, 77.0211),
    'sector 112':     (28.5761, 77.0265),
    'sector 113':     (28.5821, 77.0323),
    'sohna road':     (28.4205, 77.0455),
    'gwal pahari':    (28.4324, 77.1473),
    'manesar':        (28.3531, 76.9425),
    'dwarka expressway': (28.5186, 77.0188),
    'new':            (28.4595, 77.0266),  # generic Gurgaon centroid
}


# ─────────────────────────────────────────────────────────────
# BANGALORE NEIGHBORHOODS  →  approximate lat/lng centroids
# (data sourced from public OpenStreetMap data)
# ─────────────────────────────────────────────────────────────

BANGALORE_COORDS = {
    'whitefield':           (12.9698, 77.7500),
    'koramangala':          (12.9352, 77.6245),
    'indiranagar':          (12.9719, 77.6412),
    'hsr layout':           (12.9116, 77.6411),
    'hsr':                  (12.9116, 77.6411),
    'hebbal':               (13.0359, 77.5970),
    'bannerghatta road':    (12.8920, 77.5980),
    'bannerghatta':         (12.8920, 77.5980),
    'sarjapur road':        (12.8987, 77.6856),
    'sarjapur':             (12.8987, 77.6856),
    'electronic city':      (12.8456, 77.6603),
    'electronics city':     (12.8456, 77.6603),
    'electronic city phase 1': (12.8456, 77.6603),
    'electronics city phase 1': (12.8456, 77.6603),
    'electronic city phase 2': (12.8410, 77.6730),
    'electronics city phase 2': (12.8410, 77.6730),
    'jp nagar':             (12.9080, 77.5851),
    'jayanagar':            (12.9300, 77.5840),
    'btm layout':           (12.9166, 77.6101),
    'btm':                  (12.9166, 77.6101),
    'marathahalli':         (12.9591, 77.6974),
    'kr puram':             (13.0067, 77.6953),
    'k.r. puram':           (13.0067, 77.6953),
    'kr.puram':             (13.0067, 77.6953),
    'thanisandra':          (13.0688, 77.6190),
    'yelahanka':            (13.1007, 77.5963),
    'rajajinagar':          (12.9912, 77.5547),
    'banashankari':         (12.9248, 77.5669),
    'malleshwaram':         (13.0036, 77.5694),
    'malleswaram':          (13.0036, 77.5694),
    'hosur road':           (12.8830, 77.6500),
    'kanakapura road':      (12.8770, 77.5448),
    'mysore road':          (12.9510, 77.5040),
    'mysuru road':          (12.9510, 77.5040),
    'tumkur road':          (13.0270, 77.5180),
    'old madras road':      (13.0150, 77.6730),
    'old airport road':     (12.9715, 77.6660),
    'cv raman nagar':       (12.9856, 77.6644),
    'rt nagar':             (13.0220, 77.5934),
    'devanahalli':          (13.2476, 77.7110),
    'doddaballapur road':   (13.1320, 77.5790),
    'doddaballapur':        (13.2941, 77.5377),
    'kengeri':              (12.9081, 77.4820),
    'rajaji nagar':         (12.9912, 77.5547),
    'vijayanagar':          (12.9719, 77.5300),
    'basavanagudi':         (12.9419, 77.5750),
    'bsk':                  (12.9248, 77.5669),
    'bommanahalli':         (12.9050, 77.6210),
    'bellandur':            (12.9258, 77.6760),
    'kasavanahalli':        (12.9069, 77.7070),
    'varthur':              (12.9410, 77.7470),
    'mahadevapura':         (12.9897, 77.6957),
    'panathur':             (12.9398, 77.6988),
    'akshaya nagar':        (12.8809, 77.6378),
    'arekere':              (12.8930, 77.5900),
    'chikkalasandra':       (12.9090, 77.5447),
    'kothanur':             (13.0613, 77.6498),
    'jakkur':               (13.0791, 77.6077),
    'kalkere':              (13.0240, 77.7150),
    'horamavu':             (13.0306, 77.6634),
    'banaswadi':            (13.0152, 77.6500),
    'kammanahalli':         (13.0148, 77.6432),
    'kalyan nagar':         (13.0258, 77.6428),
    'lingarajapuram':       (13.0099, 77.6370),
    'ramamurthy nagar':     (13.0245, 77.6750),
    'nagavara':             (13.0398, 77.6228),
    'silk board':           (12.9180, 77.6228),
    'central bangalore':    (12.9716, 77.5946),
    'north bangalore':      (13.0827, 77.5870),
    'south bangalore':      (12.9134, 77.5895),
    'east bangalore':       (12.9716, 77.6940),
    'west bangalore':       (12.9716, 77.5170),
    'bangalore north':      (13.0827, 77.5870),
    'bangalore south':      (12.9134, 77.5895),
    'bangalore east':       (12.9716, 77.6940),
    'bangalore west':       (12.9716, 77.5170),
    'bangalore':            (12.9716, 77.5946),  # generic centroid
    'whitefield, bangalore': (12.9698, 77.7500),
    'hennur road':          (13.0420, 77.6440),
    'hennur':               (13.0420, 77.6440),
    'soukya road':          (12.9970, 77.7560),
    'channasandra':         (13.0320, 77.7370),
    'medahalli':            (13.0180, 77.7400),
    'budigere cross':       (13.0410, 77.7670),
    'chandapura':           (12.7984, 77.7042),
    'attibele':             (12.7867, 77.7710),
    'anekal':               (12.7088, 77.6960),
    'jigani':               (12.7783, 77.6385),
    'kanakapura':           (12.5470, 77.4172),
    'rajanukunte':           (13.1330, 77.5810),
    'rajankunte':            (13.1330, 77.5810),
    'hosakerehalli':        (12.9210, 77.5440),
    'kr pura':              (13.0067, 77.6953),
    'krishnaraja puram':    (13.0067, 77.6953),
    'mahalakshmi layout':   (13.0040, 77.5485),
    'kanakapura main road': (12.8770, 77.5448),
    'hoodi':                (12.9912, 77.7180),
    'frazer town':          (12.9990, 77.6125),
    'ulsoor':               (12.9820, 77.6201),
    'ashok nagar':          (12.9716, 77.6041),
    'shivajinagar':         (12.9870, 77.6010),
    'shivaji nagar':        (12.9870, 77.6010),
    'majestic':             (12.9774, 77.5713),
    'gandhi nagar':         (12.9783, 77.5712),
    'wilson garden':        (12.9530, 77.5996),
    'lavelle road':         (12.9719, 77.5970),
    'mg road':              (12.9756, 77.6090),
    'brigade road':         (12.9716, 77.6090),
    'commercial street':    (12.9837, 77.6094),
    'cunningham road':      (12.9888, 77.5970),
    'richmond road':        (12.9670, 77.6010),
    'uttarahalli':          (12.9050, 77.5460),
    'yelahanka new town':   (13.1007, 77.5963),
    'kogilu':               (13.1100, 77.6240),
    'bommasandra':          (12.8060, 77.7000),
    'singasandra':          (12.8810, 77.6420),
    'begur':                (12.8688, 77.6240),
    'hulimavu':             (12.8810, 77.6020),
    'gottigere':             (12.8740, 77.5920),
    'kaggalipura':           (12.7967, 77.5610),
    'kaggadasapura':         (12.9760, 77.6790),
    'mahadevpura':           (12.9897, 77.6957),
    'whitefield road':       (12.9698, 77.7500),
    'kadugodi':              (13.0026, 77.7530),
    'avalahalli':            (13.0290, 77.7550),
    'kodathi':               (12.9080, 77.7180),
    'green glen layout':     (12.9290, 77.6770),
    'haralur':               (12.9090, 77.6740),
    'haralur road':          (12.9090, 77.6740),
    'iblur':                 (12.9230, 77.6700),
    'doddanekundi':          (12.9760, 77.7000),
    'aecs layout':           (12.9760, 77.7180),
    'munnekollal':           (12.9540, 77.7090),
    'thubarahalli':          (12.9540, 77.7250),
    'brookefield':           (12.9670, 77.7180),
    'kundalahalli':          (12.9760, 77.7080),
    'sadaramangala':         (12.9890, 77.7110),
    'hoodi circle':          (12.9912, 77.7180),
}


# ─────────────────────────────────────────────────────────────
# LOOKUP WITH FUZZY MATCHING
# ─────────────────────────────────────────────────────────────

def _norm(s: str) -> str:
    """Lowercase and strip punctuation for matching."""
    return re.sub(r'[^a-z0-9 ]', ' ', str(s).lower()).strip()


def lookup(sector: str, city: str = None) -> tuple:
    """
    Return (lat, lng) for a sector string.
    Tries multiple matching strategies:
      1. Exact lookup
      2. Try removing trailing ', bangalore' / ', gurgaon'
      3. Per-token match (each comma-separated piece)
      4. Substring match on specific keys (skip generic 'bangalore' fallback)
      5. City fallback centroid (last resort only)
    Returns None if nothing matches.
    """
    if not sector or not isinstance(sector, str):
        return None

    s = _norm(sector)
    is_blr = (city == 'bangalore') if city else 'bangalore' in s
    coords = BANGALORE_COORDS if is_blr else GURGAON_COORDS

    # Generic city fallbacks — skip these in fuzzy match, use only at the end
    GENERIC = {'bangalore', 'gurgaon', 'new', 'central bangalore',
               'north bangalore', 'south bangalore',
               'east bangalore', 'west bangalore',
               'bangalore north', 'bangalore south',
               'bangalore east',  'bangalore west'}

    # 1. Exact normalized match
    if s in coords:
        return coords[s]

    # 2. Try the original raw string
    if sector.lower() in coords:
        return coords[sector.lower()]

    # 3. Try removing trailing ", bangalore" / ", gurgaon"
    trimmed = re.sub(r',\s*(bangalore|gurgaon|gurugram).*$', '', s).strip()
    if trimmed and trimmed in coords and trimmed not in GENERIC:
        return coords[trimmed]

    # 4. Per-token match — each comma-separated piece (specific keys first)
    tokens = [p.strip() for p in s.split(',') if p.strip()]
    for piece in tokens:
        if piece in coords and piece not in GENERIC:
            return coords[piece]

    # 5. Substring match on non-generic keys (longest first → more specific)
    specific_keys = [k for k in coords.keys() if k not in GENERIC]
    for key in sorted(specific_keys, key=len, reverse=True):
        if key in s:
            return coords[key]

    # 6. Looser per-token substring match (still skipping generic)
    for piece in tokens:
        for key in specific_keys:
            if key in piece or piece in key:
                return coords[key]

    # 7. City fallback (so it shows on map at least)
    if is_blr:
        return BANGALORE_COORDS['bangalore']
    return GURGAON_COORDS['new']


def add_coords(df, sector_col='sector', city_col='city'):
    """
    Adds 'lat' and 'lon' columns to df (or 'lng' alias).
    Returns a new dataframe.
    """
    import pandas as pd
    coords = df.apply(
        lambda r: lookup(r[sector_col], r[city_col] if city_col in r else None),
        axis=1,
    )
    out = df.copy()
    out['lat'] = [c[0] if c else None for c in coords]
    out['lon'] = [c[1] if c else None for c in coords]
    return out
