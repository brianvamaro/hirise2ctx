"""Bounded OpenAlex queries for the F-review literature sweep (paper-lookup skill).

~10 search calls against api.openalex.org (mailto polite pool), printing
year / citations / title / DOI for the top hits per topic. No pagination.
"""
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

QUERIES = {
    "murray_ctx_mosaic": ("global blended CTX mosaic Mars", 5, None),
    "mars_photometric_norm": ("Mars photometric normalization Minnaert Hapke correction", 7, None),
    "lunar_lambert": ("photometric normalization planetary images lunar-Lambert", 5, None),
    "rrn_mosaic": ("relative radiometric normalization image mosaic", 7, "cited_by_count:desc"),
    "dodging_leveling": ("radiometric block adjustment aerial image mosaic color balancing", 7,
                          "cited_by_count:desc"),
    "destriping": ("destriping satellite imagery moment matching", 6, "cited_by_count:desc"),
    "da_remote_sensing": ("unsupervised domain adaptation deep learning remote sensing "
                          "cross-sensor", 7, "cited_by_count:desc"),
    "consistency_overlap": ("consistency regularization overlapping images segmentation "
                            "remote sensing invariance", 6, None),
    "prediction_harmonization": ("harmonization prediction maps across image boundaries "
                                 "mosaicking deep learning artifacts", 6, None),
    "ctx_calibration": ("Context Camera CTX Mars Reconnaissance Orbiter calibration", 5, None),
}


def q(search: str, n: int, sort: str | None):
    params = {"search": search, "per_page": n, "mailto": MAILTO,
              "select": "doi,title,publication_year,cited_by_count"}
    if sort:
        params["sort"] = sort
    url = "https://api.openalex.org/works?" + urllib.parse.urlencode(params)
    with urllib.request.urlopen(url, timeout=60) as r:
        return json.load(r)["results"]


def main() -> None:
    for label, (search, n, sort) in QUERIES.items():
        print(f"\n=== {label}: {search!r}" + (f"  [{sort}]" if sort else ""))
        try:
            for w in q(search, n, sort):
                print(f"  {w['publication_year']}  c={w['cited_by_count']:>5}  "
                      f"{(w['title'] or '')[:95]}")
                print(f"      {w['doi']}")
        except Exception as e:
            print(f"  ERROR: {e}")


if __name__ == "__main__":
    main()
