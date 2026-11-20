"""Audit C1/C2 DAS geometry claims from directions npz/json only (E17, E18 RS L18) + 16k L18 decoder."""
import sys; sys.path.insert(0, __file__.rsplit("/", 1)[0])
from audit_lib import *
import itertools, torch
from scipy.stats import spearmanr
z, j = directions("E17")["add_chain_L18_L18_16k_l074"]
das = sorted(k for k in z.files if k.startswith("das"))
acc = {k: j[k]["meta"]["train_interchange_acc"] for k in das}
print("== train acc"); [print(f"  {k:18s} {acc[k]:.3f}") for k in das]
r12 = [k for k in das if parse_das(k)["k"] in (1, 2)]; r48 = [k for k in das if parse_das(k)["k"] in (4, 8)]
print(f"rank1/2 fits {len(r12)} max acc {max(acc[k] for k in r12):.3f} ({max(r12, key=acc.get)}); any >=0.5: {any(acc[k]>=.5 for k in r12)}")
print(f"rank4/8 fits {len(r48)} min acc {min(acc[k] for k in r48):.3f}; all >=0.5: {all(acc[k]>=.5 for k in r48)}; max {max(acc[k] for k in r48):.3f} ({max(r48,key=acc.get)})")
# init check: parametrized init vs plain QR
g = torch.Generator().manual_seed(0); q0 = torch.linalg.qr(torch.randn(2304, 4, generator=g))[0].double().numpy()
print("init: max|param - QR| (k=4,s=0) =", np.abs(das_init(2304, 4, 0) - q0).max())
# init capture
print("== init capture (share of energy in own init span)")
caps = {k: learned(z[k], k)[1] for k in das}
for k in das: print(f"  {k:18s} {caps[k]:.3f}")
three = [k for k in das if parse_das(k)["epochs"] == 3]
print(f"3-epoch fits init capture range {min(caps[k] for k in three):.3f}-{max(caps[k] for k in three):.3f}")
# learned rank-1 cross-seed cosines
for rk in (1, 2, 4, 8):
    fits = [k for k in das if parse_das(k)["k"] == rk and parse_das(k)["lr"] == 1e-2]
    pr = [(a, b) for a, b in itertools.combinations(fits, 2) if parse_das(a)["seed"] != parse_das(b)["seed"]]
    cl = [pcos(learned(z[a], a)[0], learned(z[b], b)[0])[0] for a, b in pr]
    raw = [pcos(z[a], z[b])[0] for a, b in pr]
    print(f"rank {rk}: fits {len(fits)}, diff-seed pairs {len(pr)}, learned top cos median {np.median(cl):.3f} range {min(cl):.3f}-{max(cl):.3f}; raw median {np.median(raw):.3f}")
    if rk == 1:
        fits2 = [k for k in das if parse_das(k)["k"] == 1]
        pr2 = [(a, b) for a, b in itertools.combinations(fits2, 2) if parse_das(a)["seed"] != parse_das(b)["seed"]]
        cl2 = [pcos(learned(z[a], a)[0], learned(z[b], b)[0])[0] for a, b in pr2]
        print(f"   incl lr3e-2: {len(pr2)} pairs median {np.median(cl2):.3f} range {min(cl2):.3f}-{max(cl2):.3f}")
# Haar p99 for k=1 top cosine (sanity)
rng = np.random.default_rng(0)
h = [abs(orth(rng.standard_normal((2304, 1)))[:, 0] @ orth(rng.standard_normal((2304, 1)))[:, 0]) for _ in range(2000)]
print("Haar p99 k=1 top cos (MC 2000):", round(np.percentile(h, 99), 3))
# error slice: learned comp captured in pca_pooled 32
P = z["pca_pooled"]
print("== learned component capture in pca_pooled[:, :32] (Haar 32/2304 = %.4f)" % (32 / 2304))
for k in das: print(f"  {k:18s} {capture(learned(z[k], k)[0], P[:, :32]):.3f}")
# decoder atoms
W = np.load(f"{SP}/deep/hf/models--google--gemma-scope-2b-pt-res/snapshots/fd571b47c1c64851e9b1989792367b9babb4af63/layer_18/width_16k/average_l0_74/params.npz")["W_dec"].astype(np.float64)
D = W / np.linalg.norm(W, axis=1, keepdims=True)
REC = [15376, 7686, 8352, 11219, 13950, 411, 14691]
z18, j18 = directions("E18")["add_chain_L18_L18_16k_l074"]
fits = {k: z[k] for k in das}; fits["das16(E18)"] = z18["das16"]
for k in ("das8", "das4"):
    print(f"E18 {k} == E17 {k}:", np.array_equal(z18[k], z[k]))
def omp(Q, m=20):
    tot = (Q ** 2).sum(); res = Q.copy(); sel = []
    for _ in range(m):
        sc = ((D @ res) ** 2).sum(1); sc[sel] = -1; sel.append(int(np.argmax(sc)))
        B = D[sel].T; coef, *_ = np.linalg.lstsq(B, Q, rcond=None); res = Q - B @ coef
    return 1 - (res ** 2).sum() / tot
print("== recurrent atoms: rank by cos^2 with learned span (0-based) ; OMP-20 share of learned comp")
rows = []
for k, U in fits.items():
    nm = "das16" if k.startswith("das16") else k
    L = learned(U, nm)[0]
    sh = ((D @ L) ** 2).sum(1); order = np.argsort(-sh); rank = np.empty_like(order); rank[order] = np.arange(len(order))
    rr = {a: int(rank[a]) for a in REC}
    o20 = omp(L)
    rows.append(dict(fit=k, k=parse_das(nm)["k"], all7_top10=all(v < 10 for v in rr.values()), omp20=round(o20, 3), **{str(a): v for a, v in rr.items()}))
R = pd.DataFrame(rows); print(R.to_string(index=False)); R.to_csv(f"{OUT}/das_atoms.csv", index=False)
# OMP null
nl = {}
for kk in (1, 2, 4, 8, 16):
    nl[kk] = np.percentile([omp(orth(rng.standard_normal((2304, kk)))) for _ in range(20)], 99)
print("OMP-20 Haar p99 (20 draws):", {k: round(v, 3) for k, v in nl.items()})
# Spearman train acc vs test flip computed in audit_das_flips.py
