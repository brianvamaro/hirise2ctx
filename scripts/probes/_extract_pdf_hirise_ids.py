"""Extract HiRISE observation IDs (ESP/PSP_######_####) from a PDF, with
surrounding context so the site/usage of each ID is identifiable.

Usage: python _extract_pdf_hirise_ids.py <pdf_path> [context_chars]
"""
import re
import sys

from pypdf import PdfReader


def main() -> int:
    path = sys.argv[1]
    ctx = int(sys.argv[2]) if len(sys.argv) > 2 else 80
    text = " ".join(p.extract_text() or "" for p in PdfReader(path).pages)
    text = re.sub(r"\s+", " ", text)
    seen = {}
    for m in re.finditer(r"[EP]SP_\d{6}_\d{4}", text):
        seen.setdefault(m.group(0), text[max(0, m.start() - ctx):m.end() + ctx])
    print(f"{len(seen)} unique HiRISE IDs in {path}\n")
    for k in sorted(seen):
        print(f"{k}\n    ...{seen[k]}...\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
