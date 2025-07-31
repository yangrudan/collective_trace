import re
from pathlib import Path
from typing import List, Dict

def parse_trace(path: str) -> List[Dict[str, str]]:
    pat = re.compile(
        r"\[TRACE\] global rank (?P<rank>\d+) in (?P<group>\S+) - "
        r"(?P<op>\w+) - "
        r"async:(?P<async>\d+), "
        r"Size: (?P<size>\d+\.\d+) MB, "
        r"Shape: (?P<shape>\([^)]+\)),"
        r"Dtype: (?P<dtype>[^,]+), "
        r"Duration: (?P<dur>\d+(?:\.\d+)?) ms, "
        r"GROUP size (?P<gs>\d+)  = (?P<ranks>\[[\d, ]+\])"
    )

    out = []
    for ln in Path(path).read_text().splitlines():
        m = pat.match(ln.strip())
        if m:
            out.append(m.groupdict())
    return out


if __name__ == "__main__":
    for rec in parse_trace("collective.txt")[:]:
        print(rec)