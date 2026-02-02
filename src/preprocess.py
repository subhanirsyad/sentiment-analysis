import re

def clean_text(text: str) -> str:
    """Normalisasi teks (sesuai notebook): lowercase, hapus URL, non-alnum, rapikan spasi."""
    s = str(text).lower()
    s = re.sub(r"http\S+|www\.\S+", " ", s)        # url
    s = re.sub(r"[^a-z0-9\s]", " ", s)              # non-alnum
    s = re.sub(r"\s+", " ", s).strip()
    return s
