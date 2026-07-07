# -*- coding: utf-8 -*-
"""
prediction.py

Thin wrapper around the Kronos model: loads the tokenizer/model once and
runs a Monte-Carlo ensemble of independent stochastic forecasts (Kronos
returns only the sample-averaged path per call, so a distribution — needed
for confidence bands and the ensemble bullish/bearish signal — requires
repeated independent calls).
"""

import sys
import numpy as np
import pandas as pd

sys.path.append("../")  # repo root, resolved relative to CWD (run from examples/)
from model import Kronos, KronosTokenizer, KronosPredictor  # noqa: E402


def load_predictor(tokenizer_name: str, model_name: str, device: str, max_context: int):
    print(f"Loading Kronos tokenizer:{tokenizer_name} model:{model_name} ...")
    tokenizer = KronosTokenizer.from_pretrained(tokenizer_name)
    model = Kronos.from_pretrained(model_name)
    return KronosPredictor(model, tokenizer, device=device, max_context=max_context)


def run_ensemble_forecast(predictor, x_df, x_timestamp, y_timestamp, pred_len, T, top_p,
                           ensemble_size):
    """Runs `ensemble_size` independent stochastic Kronos forecasts, returns stacked close paths
    with shape (ensemble_size, pred_len)."""
    close_paths = []
    for i in range(ensemble_size):
        print(f"  ensemble run {i + 1}/{ensemble_size} ...")
        pred_df = predictor.predict(
            df=x_df, x_timestamp=pd.Series(x_timestamp), y_timestamp=pd.Series(y_timestamp),
            pred_len=pred_len, T=T, top_p=top_p, sample_count=1, verbose=False,
        )
        close_paths.append(pred_df["close"].values)
    return np.array(close_paths)
