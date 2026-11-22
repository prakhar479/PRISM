"""Breadth pilot task variants (docs/PILOT_BREADTH_PLAN.md): the same templates as existing families, other counterfactuals.

  people_cities (P1): pc_en (content; harness/langfamilies.py) and its variants pc_en_swap (binding swap: the queried person's
      city is exchanged with another person's, so A and B contain the same tokens) and pc_en_query (another person is asked).
  entity_track (P11): *_dest variants of the E3 families ot_placed / ot_bin / ot_drawer (7 box labels each: one prompt set needs
      one candidate count): both runs move the queried object at the same step to different boxes, so no answer is fixed by
      the unpatched prefix.
"""
from __future__ import annotations

import dataclasses
from functools import lru_cache
from typing import Dict


@lru_cache(maxsize=1)
def families() -> Dict[str, dict]:
    from harness import families as F
    from harness import langfamilies as LF
    pc = LF.PC["pc_en"]
    out_pc = {fid: dataclasses.replace(pc, id=fid, cf=cf) for fid, cf in (("pc_en_swap", "swap"), ("pc_en_query", "query"))}
    base_ot = {k: F.OT_FAMILIES[k] for k in ("ot_placed", "ot_bin", "ot_drawer")}
    out_ot = {f"{k}_dest": dataclasses.replace(v, id=f"{k}_dest", cf="dest") for k, v in base_ot.items()}
    return {"people_cities": out_pc, "entity_track": out_ot}


P1_FAMILIES = ("pc_en", "pc_en_swap", "pc_en_query")
P11_FAMILIES = ("ot_placed_dest", "ot_bin_dest", "ot_drawer_dest")
