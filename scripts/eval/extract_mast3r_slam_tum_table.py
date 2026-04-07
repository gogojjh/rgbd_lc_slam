from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import pypdf


@dataclass(frozen=True)
class TumTable:
    columns: list[str]
    rows: dict[str, list[str]]  # method -> tokens (len=10, incl avg)


def extract_tum_table_text(pdf_path: Path) -> str:
    reader = pypdf.PdfReader(str(pdf_path))
    text = "\n".join([(p.extract_text() or "") for p in reader.pages])
    start = text.find("TUM RGB-D")
    if start == -1:
        raise ValueError("Could not find 'TUM RGB-D' in PDF")

    # Heuristic: stop at next 'Table 2' or '7-Scenes'
    end = text.find("Table 2.", start)
    if end == -1:
        end = text.find("7-Scenes", start)
    if end == -1:
        end = start + 4000

    return text[start:end]


def parse_tum_table(block: str) -> TumTable:
    lines = [" ".join(l.split()) for l in block.splitlines()]

    # Find header line containing sequence names.
    header = None
    for l in lines:
        if re.search(r"\bdesk\b", l) and re.search(r"\broom\b", l) and re.search(
            r"\bavg\b", l
        ):
            header = l
            break
    if header is None:
        raise ValueError("Could not find TUM table header")

    header_tokens = header.split()
    # Expect: 360 desk desk2 floor plant room rpy teddy xyz avg
    columns = header_tokens

    rows: dict[str, list[str]] = {}
    for l in lines:
        if l.startswith("TUM RGB-D") or l.startswith("Calibrated"):
            continue
        if not l or l.startswith("Figure") or l.startswith("Table"):
            continue

        toks = l.split()
        if len(toks) < len(columns) + 1:
            continue

        tail = toks[-len(columns) :]
        if not all(re.fullmatch(r"X|-|\d+\.\d+", x) for x in tail):
            continue

        method = " ".join(toks[: -len(columns)])
        rows[method] = tail

    if not rows:
        raise ValueError("Parsed 0 rows from TUM table block")

    return TumTable(columns=columns, rows=rows)


def main() -> None:
    pdf = Path("/tmp/mast3r_slam/2412.12392.pdf")
    block = extract_tum_table_text(pdf)
    table = parse_tum_table(block)

    print("COLUMNS:", table.columns)
    for method, vals in table.rows.items():
        print(method, ",".join(vals))


if __name__ == "__main__":
    main()
