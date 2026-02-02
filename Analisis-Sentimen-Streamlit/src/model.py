from __future__ import annotations

import joblib
from pathlib import Path
from typing import Iterable, List, Tuple

from .preprocess import clean_text

import sys
import types
import numpy as _np
import numpy.core as _np_core
import numpy.core.multiarray as _np_multiarray

def _ensure_numpy_pickle_compat():
    """Compat shim for joblib/pickle files that refer to `numpy._core.*`.

    Beberapa environment menyimpan pickle dengan modul `numpy._core`.
    Di NumPy lain, modul ini bisa bernama `numpy.core`.
    Shim ini membuat alias di `sys.modules` agar `joblib.load()` tetap bisa berjalan.
    """
    # Buat modul palsu `numpy._core` (sebagai package) agar import di pickle berhasil.
    fake_pkg = types.ModuleType("numpy._core")
    fake_pkg.multiarray = _np_multiarray  # attribute access
    # Tandai sebagai "package" (punya __path__)
    fake_pkg.__path__ = []  # type: ignore[attr-defined]

    sys.modules.setdefault("numpy._core", fake_pkg)
    sys.modules.setdefault("numpy._core.multiarray", _np_multiarray)

DEFAULT_MODEL_PATH = Path(__file__).resolve().parents[1] / "models" / "svm_tfidf.joblib"

class SentimentModel:
    """Wrapper kecil untuk load + inference model pipeline (TF-IDF + LinearSVC)."""

    def __init__(self, model_path: Path = DEFAULT_MODEL_PATH):
        self.model_path = Path(model_path)
        self._model = None

    def load(self):
        if self._model is None:
            _ensure_numpy_pickle_compat()
            self._model = joblib.load(self.model_path)
        return self

    @property
    def model(self):
        if self._model is None:
            self.load()
        return self._model

    def predict_one(self, text: str) -> Tuple[str, float]:
        """Prediksi 1 teks. Mengembalikan (label, score_margin).

        Catatan: LinearSVC tidak punya probabilitas bawaan. Kita pakai decision_function sebagai 'margin'
        (nilai makin besar -> makin yakin ke kelas tertentu).
        """
        cleaned = clean_text(text)
        label = self.model.predict([cleaned])[0]
        # decision_function bisa bentuk (n_samples,) untuk binary atau (n_samples, n_classes)
        try:
            margin = float(self.model.decision_function([cleaned])[0])
        except Exception:
            margin = float("nan")
        return label, margin

    def predict_many(self, texts: Iterable[str]) -> List[str]:
        cleaned = [clean_text(t) for t in texts]
        return self.model.predict(cleaned).tolist()
