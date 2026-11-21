"""Telemetri filtresi: {sütun: [(işleç, değer), ...]} -> filtreden geçen (clip, seg) anahtarları.

Örnek: {"altitude": [(">=", 90), ("<", 110)], "gimbal_pitch": [("<=", -40)]}
Koşullar VE ile birleşir. Telemetrisi olmayan segmentler (NaN) hiçbir koşulu sağlamaz.
"""
import duckdb
import pandas as pd

from monet.config import OUTPUT_DIR

SEGMENTS = OUTPUT_DIR / "segments.parquet"

# İzinli listeler: sütun ve işleç SQL metnine girdiği için yalnızca bunlar kabul edilir
ALLOWED_COLUMNS = {"altitude", "gimbal_pitch", "sun_elev"}
ALLOWED_OPS = {"<", "<=", ">", ">=", "="}


def build_where(filters: dict) -> tuple[str, list]:
    """Filtreyi SQL WHERE metnine ve parametre listesine çevirir; değerler metne yazılmaz (? ile)."""
    parts, params = [], []
    for col, conds in filters.items():
        if col not in ALLOWED_COLUMNS:
            raise ValueError(f"Filtrelenemeyen sütun: {col}")
        for op, value in conds:
            if op not in ALLOWED_OPS:
                raise ValueError(f"Geçersiz işleç: {op}")
            parts.append(f"{col} {op} ?")
            params.append(float(value))          # sayı değilse burada hata verir
    if not parts:
        raise ValueError("Boş filtre")
    return " AND ".join(parts), params


def filter_keys(filters: dict) -> pd.DataFrame:
    """Filtreden geçen segmentlerin (clip, seg) anahtarları. Ölçekte aynı WHERE Iceberg tablosunda çalışır."""
    where, params = build_where(filters)
    sql = f"SELECT clip, seg FROM '{SEGMENTS.as_posix()}' WHERE {where}"
    rows = duckdb.sql(sql, params=params).fetchall()
    return pd.DataFrame(rows, columns=["clip", "seg"])