"""Round 2: targeted OpenAlex queries (distinctive phrases) + Dickson 2024 abstract."""
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
    "rrn_classic": "radiometric normalization multitemporal high resolution satellite images",
    "irmad": "automatic radiometric normalization multivariate alteration detection",
    "color_balance_mosaic": "color correction optimization image stitching mosaic seam",
    "wallis_dodging": "Wallis filter dodging uneven illumination aerial images",
    "moment_matching": "destriping moment matching histogram MODIS",
    "da_overview": "domain adaptation classification remote sensing data overview",
    "deep_coral": "Deep CORAL correlation alignment domain adaptation",
    "tile_artifact_cnn": "boundary artifacts convolutional segmentation large images tiling blending",
    "bickel_rockfall": "rockfall detection deep learning planetary images Mars Moon",
    "incidence_invariance_planetary": "illumination invariant crater detection incidence angle deep learning",
    "hapke_vs_minnaert_mars": "Mars surface photometry CTX HRSC atmospheric correction",
}


def q(search: str, n: int = 6):
    params = {"search": search, "per_page": n, "mailto": MAILTO,
              "select": "doi,title,publication_year,cited_by_count"}
    url = "https://api.openalex.org/works?" + urllib.parse.urlencode(params)
    with urllib.request.urlopen(url, timeout=60) as r:
        return json.load(r)["results"]


def abstract_of(doi: str) -> str:
    url = (f"https://api.openalex.org/works/doi:{urllib.parse.quote(doi, safe='')}"
           f"?select=abstract_inverted_index,title&mailto={MAILTO}")
    with urllib.request.urlopen(url, timeout=60) as r:
        d = json.load(r)
    inv = d.get("abstract_inverted_index") or {}
    pos = {i: w for w, idxs in inv.items() for i in idxs}
    return d["title"] + "\n" + " ".join(pos[i] for i in sorted(pos))


def main() -> None:
    for label, search in QUERIES.items():
        print(f"\n=== {label}: {search!r}")
        try:
            for w in q(search):
                print(f"  {w['publication_year']}  c={w['cited_by_count']:>5}  "
                      f"{(w['title'] or '')[:95]}")
                print(f"      {w['doi']}")
        except Exception as e:
            print(f"  ERROR: {e}")

    print("\n\n===== Dickson et al. 2024 (Murray CTX mosaic) abstract =====")
    try:
        print(abstract_of("10.1029/2024ea003555"))
    except Exception as e:
        print(f"ERROR: {e}")


if __name__ == "__main__":
    main()
