# src/utils/data_loader.py
import pandas as pd
from pathlib import Path

DATA_RAW = Path(__file__).resolve().parents[2] / "data" / "raw"


def load_maestra_articulos():
    return pd.read_csv(DATA_RAW / "maestra_articulos.csv")


def load_linea_tickets():
    return pd.read_csv(DATA_RAW / "linea_tickets.csv")