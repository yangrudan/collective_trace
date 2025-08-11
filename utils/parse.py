"""
Test trace parsing
"""
from pathlib import Path
from typing import List, Dict
from pretty_print import parse_pat

def parse_trace(path: str) -> List[Dict[str, str]]:
    """Parse a collective trace log and return a list of records."""
    pat = parse_pat()

    out = []
    for ln in Path(path).read_text(encoding="utf-8").splitlines():
        m = pat.match(ln.strip())
        if m:
            out.append(m.groupdict())
    return out


if __name__ == "__main__":
    for rec in parse_trace("collective.txt")[:]:
        print(rec)
