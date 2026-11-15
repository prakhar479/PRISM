#!/usr/bin/env python
"""Record that a human reviewed the template packs of some languages (data/lang/signoff.json).

    python scripts/sign_templates.py --langs hi,fr --by "Your Name"       # after reading data/lang/templates_review.md
    python scripts/sign_templates.py --list                                # who signed what, and whether it is still valid
    python scripts/sign_templates.py --revoke hi

Only a person should run this: it is the record that a native speaker checked grammar and naturalness. Each entry stores the
pack's content hash, so editing a template (harness/langfamilies.py) voids that sign-off until it is signed again. Rebuild the
prompt sets (scripts/build_lang_tasks.py) after any edit; E13 stamps `validated` from this file when it runs.
"""
import argparse
import json
import sys
import time

import _bootstrap  # noqa: F401
from harness import langfamilies as LF


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--langs", default="")
    ap.add_argument("--by", default="")
    ap.add_argument("--note", default="")
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--revoke", default="")
    a = ap.parse_args()
    cur = LF.signoffs()
    if a.list or not (a.langs or a.revoke):
        for fid in sorted(LF.LANG_OF):
            e = cur.get(fid)
            state = "not signed" if not e else ("VALID" if LF.is_validated(fid) else "VOID (template edited since)")
            print(f"{fid:12s} {state:30s} {('by ' + e['by'] + ' on ' + e['date']) if e else ''}")
        return 0
    if a.revoke:
        for lang in a.revoke.split(","):
            for t in LF.FAMILIES:
                cur.pop(LF.family_id(t, lang), None)
    if a.langs:
        if not a.by.strip():
            ap.error("--by \"Your Name\" is required: the sign-off records who reviewed")
        bad = [l for l in a.langs.split(",") if l not in LF.LANGUAGES]
        if bad:
            ap.error(f"unknown languages {bad}; known {list(LF.LANGUAGES)}")
        for lang in a.langs.split(","):
            for t in LF.FAMILIES:
                fid = LF.family_id(t, lang)
                cur[fid] = {"by": a.by.strip(), "date": time.strftime("%Y-%m-%d"), "pack_hash": LF.pack_hash(fid), "note": a.note}
    LF.SIGNOFF_FILE.write_text(json.dumps(cur, indent=1, sort_keys=True) + "\n")
    print(f"wrote {LF.SIGNOFF_FILE}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
