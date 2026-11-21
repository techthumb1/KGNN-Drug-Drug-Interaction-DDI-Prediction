"""Locally cached NLM RxTerms names; no patient search terms leave the server."""
import csv
import io
import logging
import os
import re
from zipfile import ZipFile

from src.utils.paths import DRUGBANK_DATA_DIR

RXTERMS_RELEASE = "202611"
RXTERMS_URL = f"https://data.lhncbc.nlm.nih.gov/public/rxterms/release/RxTerms{RXTERMS_RELEASE}.zip"
RXTERMS_PATH = DRUGBANK_DATA_DIR / "rxterms.zip"


def search_text(value):
    return " ".join(re.findall(r"[^\W_]+", str(value).casefold(), re.UNICODE))


def load_products():
    """Fetch a public vocabulary once, then search it offline on subsequent runs."""
    try:
        from src.utils.data_loader import download_file
        download_file(os.getenv("RXTERMS_URL", RXTERMS_URL), RXTERMS_PATH)
        with ZipFile(RXTERMS_PATH) as archive:
            def rows(pattern):
                name = next(n for n in archive.namelist() if re.fullmatch(pattern, n))
                return csv.DictReader(io.StringIO(archive.read(name).decode("utf-8-sig")), delimiter="|")
            ingredients = {}
            for row in rows(r"RxTermsIngredients\d+\.txt"):
                ingredients.setdefault(row["RXCUI"], {})[row["ING_RXCUI"]] = row["INGREDIENT"]
            products = []
            for row in rows(r"RxTerms\d+\.txt"):
                if row["IS_RETIRED"] or row["SUPPRESS_FOR"]:
                    continue
                names = [row[k] for k in ("FULL_NAME", "PSN", "DISPLAY_NAME", "BRAND_NAME", "DISPLAY_NAME_SYNONYM") if row[k]]
                parts = list(ingredients.get(row["RXCUI"], {}).values())
                products.append((row["RXCUI"], row["PSN"] or row["FULL_NAME"], parts,
                                 search_text(" ".join(names + parts))))
            if not products:
                raise ValueError("RxTerms contains no active medication names")
            return products, "ready"
    except Exception as error:
        logging.getLogger(__name__).warning("RxTerms name coverage unavailable: %s", error)
        return [], "unavailable"
