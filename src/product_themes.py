"""Strategic product-name themes used for cluster profiling.

These themes are not the segmentation itself. They are an editable validation
overlay that helps explain whether discovered product-led clusters map to
commercially useful needs such as baby, pet, free-from, fitness, or organic.
"""
from __future__ import annotations

import re
import unicodedata


STRATEGIC_THEME_PATTERNS: dict[str, tuple[str, ...]] = {
    "baby": (
        r"\bBEBE\b",
        r"\bINFANTIL\b",
        r"\bPOTITO[S]?\b",
        r"\bPAPILLA[S]?\b",
        r"\bPANAL(?:ES)?\b",
        r"\bTOALLITA[S]?\b",
        r"\bLECHE INFANTIL\b",
    ),
    "pet": (
        r"\bMASCOTA[S]?\b",
        r"\bPERRO[S]?\b",
        r"\bGATO[S]?\b",
        r"\bPIENSO\b",
        r"\bCOMIDA (?:PERRO|GATO)\b",
        r"\bARENA GATO\b",
    ),
    "organic_bio": (
        r"\bBIO\b",
        r"\bECO\b",
        r"\bECOLOGIC[OA]S?\b",
        r"\bORGANIC[OA]S?\b",
    ),
    "gluten_free": (
        r"\bSIN GLUTEN\b",
        r"\bS/GLUTEN\b",
        r"\bGLUTEN FREE\b",
        r"\bCELIAC[OA]S?\b",
    ),
    "lactose_free": (
        r"\bSIN LACTOSA\b",
        r"\bS/LACTOSA\b",
        r"\bLACTOSA FREE\b",
    ),
    "protein_fitness": (
        r"\bPROTEIN[A]?\b",
        r"\bPROTEICO[S]?\b",
        r"\bWHEY\b",
        r"\bFITNESS\b",
        r"\bISOTONIC[OA]S?\b",
    ),
    "health_wellness": (
        r"\bINTEGRAL(?:ES)?\b",
        r"\bSIN AZUCAR(?:ES)?\b",
        r"\bS/AZUCAR\b",
        r"\b0\s*%",
        r"\bLIGHT\b",
        r"\bBAJO EN\b",
        r"\bDESNATAD[OA]S?\b",
        r"\bAVENA\b",
        r"\bCHIA\b",
        r"\bQUINOA\b",
    ),
    "plant_based": (
        r"\bVEGAN[OA]S?\b",
        r"\bVEGETARIAN[OA]S?\b",
        r"\bVEGETAL(?:ES)?\b",
        r"\bTOFU\b",
        r"\bSEITAN\b",
        r"\bSOJA\b",
    ),
    "ethnic_international": (
        r"\bMEXICAN[OA]S?\b",
        r"\bTEX MEX\b",
        r"\bASIATIC[OA]S?\b",
        r"\bCHIN[OA]S?\b",
        r"\bJAPONES(?:A|AS|ES)?\b",
        r"\bINDI[OA]S?\b",
        r"\bARABE[S]?\b",
        r"\bORIENTAL(?:ES)?\b",
        r"\bMARROQUI(?:ES)?\b",
        r"\bSUSHI\b",
        r"\bCURRY\b",
        r"\bCOUSCOUS\b",
        r"\bKEBAB\b",
        r"\bTACO[S]?\b",
        r"\bNACHO[S]?\b",
        r"\bWRAP[S]?\b",
    ),
    "fresh": (
        r"\bFRESC[OA]S?\b",
        r"\bFRUTA[S]?\b",
        r"\bVERDURA[S]?\b",
        r"\bHORTALIZA[S]?\b",
        r"\bENSALADA[S]?\b",
        r"\bPESCADO[S]?\b",
        r"\bCARNE[S]?\b",
    ),
    "ready_meals": (
        r"\bPLATO[S]? PREPARAD[OA]S?\b",
        r"\bPREPARAD[OA]S?\b",
        r"\bLISTO PARA\b",
        r"\bMICROONDAS\b",
        r"\bPIZZA[S]?\b",
        r"\bLASANA[S]?\b",
    ),
    "premium_indulgence": (
        r"\bGOURMET\b",
        r"\bPREMIUM\b",
        r"\bRESERVA\b",
        r"\bIBERIC[OA]S?\b",
        r"\bFOIE\b",
    ),
}


def normalize_product_text(text: str | None) -> str:
    """Uppercase, accent-insensitive product text for regex matching."""
    if text is None:
        return ""
    normalized = unicodedata.normalize("NFKD", str(text))
    ascii_text = normalized.encode("ascii", "ignore").decode("ascii")
    return re.sub(r"\s+", " ", ascii_text.upper()).strip()


_COMPILED_THEME_PATTERNS = {
    theme: tuple(re.compile(pattern) for pattern in patterns)
    for theme, patterns in STRATEGIC_THEME_PATTERNS.items()
}


def classify_product_themes(description: str | None) -> list[str]:
    """Return all strategic themes matched by a product description."""
    text = normalize_product_text(description)
    if not text:
        return []
    return [
        theme
        for theme, patterns in _COMPILED_THEME_PATTERNS.items()
        if any(pattern.search(text) for pattern in patterns)
    ]
