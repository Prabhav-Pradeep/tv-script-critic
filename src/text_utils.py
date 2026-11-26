"""Name masking, so critics judge writing rather than which characters appear.

A token counts as a name if it appears inside a speaker label ("Gil Grissom: ...") and is
capitalised in at least CAP_SHARE of its occurrences away from the start of a line.
The second rule keeps words like "man", "doctor" or "class" (also used as labels) visible.
"""
import re
from collections import Counter

DLG = re.compile(r"^([A-Za-z][A-Za-z0-9 .'\-/&,]{0,40}?):\s+\S")
WORD = re.compile(r"[A-Za-z][A-Za-z']+")
CAP_SHARE = 0.9
MIN_COUNT = 5


def find_names(texts):
    label_tokens, cap, total = set(), Counter(), Counter()
    for t in texts:
        for line in t.split("\n"):
            m = DLG.match(line)
            if m:
                label_tokens.update(w.lower() for w in WORD.findall(m.group(1)))
                body = line[m.end(1) + 1:]
            else:
                body = line
            words = WORD.findall(body)
            for w in words[1:]:          # skip the first word: sentence-initial capitals mislead
                lw = w.lower()
                total[lw] += 1
                cap[lw] += w[0].isupper()
    return {w for w in label_tokens
            if total[w] >= MIN_COUNT and cap[w] / total[w] >= CAP_SHARE and len(w) > 1}


def mask_names(text, names, token="NAME"):
    def sub(m):
        return token if m.group(0).lower() in names else m.group(0)
    return WORD.sub(sub, text)
