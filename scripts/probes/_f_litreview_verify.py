"""Verify specific DOIs + pull two abstracts for the F review (bounded: 4 lookups)."""
import json
import sys
import urllib.parse
import urllib.request
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

import truststore

truststore.inject_into_ssl()

MAILTO = "brianvamaro@gmail.com"

DOIS = {
    "HLS (Claverie 2018?)": "10.1016/j.rse.2018.09.002",
    "BRDF c-factor (Roy 2016?)": "10.1016/j.rse.2016.01.023",
    "Bickel rockfalls 2020": "10.1109/jstars.2020.2991588",
    "Mars-from-Moon DA 2022": "10.1109/jstars.2022.3156371",
}


def work(doi: str, with_abstract: bool = False) -> dict:
    sel = "title,publication_year,cited_by_count,doi"
    if with_abstract:
        sel += ",abstract_inverted_index"
    url = (f"https://api.openalex.org/works/doi:{urllib.parse.quote(doi, safe='')}"
           f"?select={sel}&mailto={MAILTO}")
    with urllib.request.urlopen(url, timeout=60) as r:
        return json.load(r)


def abstract(d: dict) -> str:
    inv = d.get("abstract_inverted_index") or {}
    pos = {i: w for w, idxs in inv.items() for i in idxs}
    return " ".join(pos[i] for i in sorted(pos))


for label, doi in DOIS.items():
    try:
        d = work(doi, with_abstract=label.startswith(("Bickel", "Mars-from")))
        print(f"\n{label}: VERIFIED -> {d['title']} ({d['publication_year']}, "
              f"c={d['cited_by_count']})")
        if label.startswith(("Bickel", "Mars-from")):
            print("  ABSTRACT:", abstract(d)[:900])
    except Exception as e:
        print(f"\n{label}: FAILED ({e})")
