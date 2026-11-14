"""Komut satırından arama.

Tek sorgu:  python scripts/search.py "an aerial photo of a paved road with dashed lane markings"
Döngü:      python scripts/search.py           (boş satır ya da q ile çıkılır)
"""
import argparse
import time

import pandas as pd

from monet.search.engine import DEFAULT_INDEX, SearchEngine


def mmss(sec: float) -> str:
    # Video saniyesini oynatıcıda aranacak biçime çevir: 83.4 -> 01:23
    return f"{int(sec // 60):02d}:{int(sec % 60):02d}"


def show(engine: SearchEngine, query: str, k: int):
    t = time.time()
    res = engine.search(query, k=k)
    dt = time.time() - t
    out = pd.DataFrame({
        "#": res["rank"], "skor": res["score"], "video": res["clip"] + ".mp4",
        "bas": res["video_bas"].map(mmss), "son": res["video_son"].map(mmss),
        "gercek_saat_utc": res["t_bas"].dt.strftime("%Y-%m-%d %H:%M:%S"), "kare": res["n_frames"],
    })
    print(f"\n'{query}'  ({dt * 1000:.0f} ms)")
    print(out.to_string(index=False))


def main():
    p = argparse.ArgumentParser(description="Metinle segment ara")
    p.add_argument("query", nargs="?", help="sorgu metni (verilmezse döngü açılır)")
    p.add_argument("-k", type=int, default=10, help="kaç sonuç")
    p.add_argument("--index", default=DEFAULT_INDEX, help="indeks klasör adı")
    args = p.parse_args()

    t = time.time()
    engine = SearchEngine(args.index)
    print(f"indeks: {args.index} | {len(engine.vectors)} segment | acilis: {time.time() - t:.1f} sn")

    if args.query:
        show(engine, args.query, args.k)
        return
    while (q := input("\nsorgu> ").strip()) not in ("", "q"):
        show(engine, q, args.k)


if __name__ == "__main__":
    main()