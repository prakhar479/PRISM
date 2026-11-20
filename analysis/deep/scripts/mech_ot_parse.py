"""Parse object-tracking queries (pilot + E3 families): per pair, the queried object's initial box, its moves in A and B, and
the role of A's / B's answer (initial box = 'stay', or a move destination)."""
import re
from mech_common import *

PAT = {  # (init regex, move regex, query regex)
    "pilot": (r"The (\w+) is in box (\w)\.", r"The (\w+) moves to box (\w)\.", r"The (\w+) is in box$"),
    "ot_placed": (r"The (\w+) is placed in box (\w)\.", r"The (\w+) goes to box (\w)\.", r"The (\w+) is in box$"),
    "ot_bin": (r"The (\w+) sits in bin (\w)\.", r"The (\w+) is moved into bin (\w)\.", r"The (\w+) is now in bin$"),
    "ot_drawer": (r"There is an? (\w+) in drawer (\w)\.", r"Someone puts the (\w+) into drawer (\w)\.", r"So the (\w+) is in drawer$"),
}


def parse(q, fam):
    pi, pm, pq = PAT[fam]
    init = dict(re.findall(pi, q))
    moves = re.findall(pm, q)
    obj = re.search(pq, q).group(1)
    return obj, init, moves


def table(name):
    p = prompts(name)
    rows = []
    for r in p.itertuples():
        fam = r.family if r.family in PAT else "pilot"
        qa, qb = r.clean_prompt.split("\n")[-1], r.corrupt_prompt.split("\n")[-1]
        oa, ia, ma = parse(qa, fam)
        ob, ib, mb = parse(qb, fam)
        A, B = r.clean_target.strip(), r.corrupt_target.strip()
        roleA = "stay" if A == ia.get(oa) and not any(o == oa for o, _ in ma) else "moved"
        roleB = "stay" if B == ib.get(ob) and not any(o == ob for o, _ in mb) else "moved"
        nmA = sum(o == oa for o, _ in ma); nmB = sum(o == ob for o, _ in mb)
        rows.append(dict(pair_id=r.pair_id, family=r.family, obj=oa, init=ia.get(oa), A=A, B=B, roleA=roleA, roleB=roleB,
                         nmovesA=nmA, nmovesB=nmB, cstep=r.meta.get("corrupt_step"), same_query=oa == ob,
                         B_is_init_of_query=(B == ia.get(oa)), A_is_init=(A == ia.get(oa))))
    return pd.DataFrame(rows)

if __name__ == "__main__":
    for n in ["entity_track_pilot", "entity_track_e3"]:
        t = table(n)
        print(n, len(t)); print(t.groupby(["family", "roleA", "roleB", "nmovesA", "nmovesB"]).size().to_string())
