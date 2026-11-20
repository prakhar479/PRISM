"""Audit: RQ1 gate (E3 RS L18, fresh families, fit fresh, f_pilot) + red-team qualifications."""
import sys; sys.path.insert(0, __file__.rsplit("/", 1)[0])
from audit_lib import *
d = load("E3", extra=("cand_logits",))
d = d[(d.task == "add_chain") & (d.layer == 18) & (d.fit_families == "fresh") & (~d.family.astype(str).str.startswith("pilot"))]
print("families", sorted(d.family.unique()))
c = Cell(d, "f_pilot")
full = c.flips(c.get("full_A")); mp = c.flips(c.get("feat_A_meanPos")); eb = c.flips(c.get("feat_A_errB"))
e = c.elig
print("n pairs", len(c.pairs), "eligible", e.sum())
diff = (full - mp)[e]; n = len(diff)
pt, lo, hi, _ = boot(lambda i: diff[i].mean(), n, 2000)
print(f"RQ1 shortfall full_A - meanPos = {pt:.4f} [{lo:.4f}, {hi:.4f}]  full_A {np.nanmean(full):.4f} meanPos {np.nanmean(mp):.4f}")
de = (full - eb)[e]
print(f"errB shortfall = {de.mean():.4f}")
# family-cluster bootstrap
fam = c.ca.family.to_numpy()[e]; uf = np.unique(fam)
rng = np.random.default_rng(0); s = []
for _ in range(2000):
    pick = rng.choice(uf, len(uf)); s.append(np.concatenate([diff[fam == f] for f in pick]).mean())
print("cluster CI", np.percentile(s, [2.5, 97.5]).round(4), "n clusters", len(uf))
for f in uf:
    print(" family", f, (fam == f).sum(), round(diff[fam == f].mean(), 4))
# margin split: clean_A lead of A over best other candidate
cl = np.stack(c.ca.cand_logits.to_numpy())
la = c.ca.logit_A.to_numpy()
other = np.where(np.isclose(cl, la[:, None]), -np.inf, cl)
other = np.where(np.isfinite(other), other, -np.inf)
lead = la - other.max(1)
lead_e = lead[e]; med = np.median(lead_e)
lowm = lead_e <= med
print(f"median lead {med:.3f}; low-lead shortfall {diff[lowm].mean():.4f} (median {np.median(lead_e[lowm]):.2f}, n {lowm.sum()}); high {diff[~lowm].mean():.4f} (median {np.median(lead_e[~lowm]):.2f}, n {(~lowm).sum()})")
lowm2 = lead_e < med
print(f"(strict <) low {diff[lowm2].mean():.4f} high {diff[~lowm2].mean():.4f}")
