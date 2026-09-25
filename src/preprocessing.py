
import regex as re
import unicodedata
from unidecode import unidecode


NAME_ABBREVIATIONS = {
    "pvt": "private",
    "ltd": "limited",
    "corp": "corporation",
    "inc": "incorporated",
}

ADDRESS_ABBREVIATIONS = {
    "rd": "road",
    "ave": "avenue",
    "blvd": "boulevard",
    "apt": "apartment",
}


def normalize_text(text):

    if text is None:
        return ""

    text = unicodedata.normalize("NFKC", str(text))

    text = text.casefold()

    text = text.replace("&", " and ")

    # Preserve Unicode letters, combining marks and numbers
    text = re.sub(
        r"[^\p{L}\p{M}\p{N}\s]",
        " ",
        text
    )

    text = re.sub(r"\s+", " ", text).strip()

    return text


def expand_abbreviations(text, dictionary):

    return " ".join(
        dictionary.get(word, word)
        for word in text.split()
    )


def normalize_name(name):

    text = normalize_text(name)

    return expand_abbreviations(
        text,
        NAME_ABBREVIATIONS
    )


def normalize_address(address):

    text = normalize_text(address)

    return expand_abbreviations(
        text,
        ADDRESS_ABBREVIATIONS
    )


def transliterate_text(text):

    return normalize_text(
        unidecode(str(text))
    )


def preprocess_dataframe(df):

    df = df.copy()

    df["name_clean"] = df["business_name"].map(
        normalize_name
    )

    df["address_clean"] = df["business_address"].map(
        normalize_address
    )

    df["country_clean"] = df["country"].map(
        normalize_text
    )

    return df


if __name__ == "__main__":

    examples = [
        "ABC Technologies Pvt. Ltd.",
        "ABC Technologies Private Limited",
        "राम मार्केटिंग प्राइवेट लिमिटेड",
        "Société Française S.A.S.",
        "12, Anna Salai Rd."
    ]

    for example in examples:

        print("Original:", example)
        print("Normalized:", normalize_text(example))
        print("Name clean:", normalize_name(example))
        print("Transliterated:", transliterate_text(example))
        print("-" * 50)
