"""Thin wrapper around the official Kronos API (model/kronos.py).

Importing this module never touches the network. Model weights are only
downloaded from the Hugging Face Hub the first time predict() is actually
called, and are cached on the instance for subsequent calls.
"""
import os
import sys
from typing import Optional

import pandas as pd

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)


class KronosForecaster:
    def __init__(self, tokenizer_name: str = "NeoQuasar/Kronos-Tokenizer-base",
                 model_name: str = "NeoQuasar/Kronos-small",
                 max_context: int = 512, device: Optional[str] = None):
        self.tokenizer_name = tokenizer_name
        self.model_name = model_name
        self.max_context = max_context
        self.device = device
        self._predictor = None

    @property
    def is_loaded(self) -> bool:
        return self._predictor is not None

    def _ensure_loaded(self):
        if self._predictor is not None:
            return
        from model import Kronos, KronosPredictor, KronosTokenizer

        tokenizer = KronosTokenizer.from_pretrained(self.tokenizer_name)
        model = Kronos.from_pretrained(self.model_name)
        self._predictor = KronosPredictor(
            model, tokenizer, device=self.device, max_context=self.max_context
        )

    def predict(self, df: pd.DataFrame, x_timestamp: pd.Series, y_timestamp: pd.Series,
                pred_len: int, T: float = 1.0, top_k: int = 0, top_p: float = 0.9,
                sample_count: int = 1, verbose: bool = True) -> pd.DataFrame:
        self._ensure_loaded()
        return self._predictor.predict(
            df=df, x_timestamp=x_timestamp, y_timestamp=y_timestamp, pred_len=pred_len,
            T=T, top_k=top_k, top_p=top_p, sample_count=sample_count, verbose=verbose,
        )

    def predict_batch(self, df_list, x_timestamp_list, y_timestamp_list, pred_len,
                       T: float = 1.0, top_k: int = 0, top_p: float = 0.9,
                       sample_count: int = 1, verbose: bool = True):
        self._ensure_loaded()
        return self._predictor.predict_batch(
            df_list=df_list, x_timestamp_list=x_timestamp_list, y_timestamp_list=y_timestamp_list,
            pred_len=pred_len, T=T, top_k=top_k, top_p=top_p, sample_count=sample_count, verbose=verbose,
        )
