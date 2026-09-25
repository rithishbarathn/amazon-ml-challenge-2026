"""Shared paths, text normalisation and small helpers for the ER pipeline.

Every stage imports from here so that training and test data are always
normalised in exactly the same way.
"""

import os
import unicodedata
import zlib
from pathlib import Path

import regex as re
from unidecode import unidecode

# --------------------------------------------------------------------------
# Paths (override the dataset location with the ER_DATA_DIR env variable)
# --------------------------------------------------------------------------

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = Path(
    os.environ.get(
        "ER_DATA_DIR",
        ROOT / "dataset" / "student_resource" / "dataset",
    )
)
WORK_DIR = Path(os.environ.get("ER_WORK_DIR", ROOT / "work"))
OUTPUT_DIR = Path(os.environ.get("ER_OUTPUT_DIR", ROOT / "output"))

SPLITS = ("train", "test")
SOURCES = ("source1", "source2", "source3")


def raw_path(split, source):
    return DATA_DIR / split / f"{split}_{source}.tsv"


def prep_path(split, source):
    return WORK_DIR / "prep" / f"{split}_{source}.parquet"


def emb_path(split, source):
    return WORK_DIR / "emb" / f"{split}_{source}.npy"


# --------------------------------------------------------------------------
# Train-set partitioning by Source 1 id (deterministic, no RNG state needed)
#   buckets 0-5 -> encoder training, 6-8 -> matcher training, 9 -> holdout
# --------------------------------------------------------------------------

def bucket(entity_id, n=10):
    return zlib.crc32(entity_id.encode("utf-8")) % n


# --------------------------------------------------------------------------
# Text normalisation
# --------------------------------------------------------------------------

NAME_ABBREVIATIONS = {
    "pvt": "private", "pte": "private", "prvt": "private",
    "ltd": "limited", "ltda": "limited", "lmt": "limited",
    "corp": "corporation", "co": "company", "cos": "companies",
    "inc": "incorporated", "incorp": "incorporated",
    "intl": "international", "int": "international",
    "mfg": "manufacturing", "mfrs": "manufacturers",
    "svc": "services", "svcs": "services", "serv": "services",
    "assn": "association", "assoc": "associates",
    "bros": "brothers", "natl": "national", "ent": "enterprises",
    "mgmt": "management", "dev": "development", "grp": "group",
    "ind": "industries", "inds": "industries", "eng": "engineering",
    "engg": "engineering", "tech": "technologies",
    "cie": "compagnie", "ste": "societe", "sa": "sa",
}

ADDRESS_ABBREVIATIONS = {
    "st": "street", "str": "street", "rd": "road", "ave": "avenue",
    "av": "avenue", "blvd": "boulevard", "bd": "boulevard",
    "bvd": "boulevard", "dr": "drive", "ln": "lane", "hwy": "highway",
    "pkwy": "parkway", "ct": "court", "pl": "place", "cir": "circle",
    "sq": "square", "ter": "terrace", "trl": "trail", "fwy": "freeway",
    "expy": "expressway", "ste": "suite", "apt": "apartment",
    "fl": "floor", "flr": "floor", "bldg": "building", "rm": "room",
    "n": "north", "s": "south", "e": "east", "w": "west",
    "ne": "northeast", "nw": "northwest", "se": "southeast",
    "sw": "southwest", "mt": "mount", "ft": "fort", "hts": "heights",
    "nr": "near", "opp": "opposite", "stn": "station", "clny": "colony",
    "ngr": "nagar", "mkt": "market", "vill": "village", "vil": "village",
    "dist": "district", "distt": "district", "po": "post office",
    "ps": "police station", "r": "rue", "rte": "route", "ch": "chemin",
    "imp": "impasse", "fbg": "faubourg", "all": "allee", "pt": "point",
}

US_STATES = {
    "al": "alabama", "ak": "alaska", "az": "arizona", "ar": "arkansas",
    "ca": "california", "co": "colorado", "ct": "connecticut",
    "de": "delaware", "fl": "florida", "ga": "georgia", "hi": "hawaii",
    "id": "idaho", "il": "illinois", "in": "indiana", "ia": "iowa",
    "ks": "kansas", "ky": "kentucky", "la": "louisiana", "me": "maine",
    "md": "maryland", "ma": "massachusetts", "mi": "michigan",
    "mn": "minnesota", "ms": "mississippi", "mo": "missouri",
    "mt": "montana", "ne": "nebraska", "nv": "nevada",
    "nh": "new hampshire", "nj": "new jersey", "nm": "new mexico",
    "ny": "new york", "nc": "north carolina", "nd": "north dakota",
    "oh": "ohio", "ok": "oklahoma", "or": "oregon", "pa": "pennsylvania",
    "ri": "rhode island", "sc": "south carolina", "sd": "south dakota",
    "tn": "tennessee", "tx": "texas", "ut": "utah", "vt": "vermont",
    "va": "virginia", "wa": "washington", "wv": "west virginia",
    "wi": "wisconsin", "wy": "wyoming", "dc": "district of columbia",
}

# Legal-form tokens, removed to build the "core" business name.
LEGAL_TOKENS = {
    "private", "limited", "corporation", "company", "companies",
    "incorporated", "llc", "llp", "lp", "plc", "pllc", "pc", "opc",
    "the", "and", "of", "sas", "sasu", "sarl", "eurl", "sa", "sci",
    "snc", "scop", "compagnie", "societe", "fils", "et", "de", "des",
    "du", "la", "le", "les", "dba", "aka", "group", "holdings",
}

_NON_ALNUM = re.compile(r"[^a-z0-9]+")
_REPEATS = re.compile(r"(.)\1+")


def basic_normalize(text):
    """Transliterate to ASCII, lowercase and strip punctuation."""
    if not text:
        return ""
    text = unicodedata.normalize("NFKC", str(text))
    text = unidecode(text).lower().replace("&", " and ")
    return _NON_ALNUM.sub(" ", text).strip()


def _expand(tokens, mapping):
    return [mapping.get(token, token) for token in tokens]


def normalize_name(text):
    return " ".join(_expand(basic_normalize(text).split(), NAME_ABBREVIATIONS))


def normalize_address(text, country):
    tokens = basic_normalize(text).split()
    state_pos = None
    if country == "US" and tokens:
        # A US state code sits at the end ("..., Austin, TX") or just before
        # the ZIP ("..., TX 78701"). Only that position is expanded, so that
        # "FL" / "CT" elsewhere can still mean floor / court.
        pos = len(tokens) - 1
        if pos > 0 and tokens[pos].isdigit():
            pos -= 1
        if tokens[pos] in US_STATES:
            state_pos = pos
    out = []
    for i, token in enumerate(tokens):
        if i == state_pos:
            out.append(US_STATES[token])
        else:
            out.append(ADDRESS_ABBREVIATIONS.get(token, token))
    return " ".join(out)


def join_initials(tokens):
    """Merge runs of single letters ("s a s" from "S.A.S." -> "sas")."""
    out, run = [], []
    for t in tokens:
        if len(t) == 1 and t.isalpha():
            run.append(t)
            continue
        if run:
            out.append("".join(run))
            run = []
        out.append(t)
    if run:
        out.append("".join(run))
    return out


def core_name(name_norm):
    """Business name without legal-form / stop tokens."""
    tokens = [t for t in join_initials(name_norm.split()) if t not in LEGAL_TOKENS]
    return " ".join(tokens) if tokens else name_norm


def squash(text):
    """Collapse repeated characters (limittedd -> limited, aa -> a).

    Makes naive transliterations of Indic scripts closer to their Latin
    spellings. Applied identically to every record, so it is harmless for
    regular text.
    """
    return _REPEATS.sub(r"\1", text)
