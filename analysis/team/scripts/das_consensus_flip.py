"""Within rank 1 and 2 (E17 RS L18): does the share of the rotation that lies on the cross-seed consensus direction track
held-out flip? (descriptive; consensus is built from the same fits, so partly circular). Post hoc."""
import sys
sys.path.insert(0, __file__.rsplit("/", 1)[0])
from das_lib import *
import pandas as pd
from scipy.stats import spearmanr
Z, J = load("E17", "add_chain_L18_L18_16k_l074"); c1 = np.load(f"{OUT}/rs18_das1_consensus.npy")
cf = pd.read_csv(f"{OUT}/e17_configs.csv").set_index("est")
rows = []
for k in Z:
    if k.startswith("das"):
        U = orth(Z[k]); rows.append(dict(est=k, k=U.shape[1], cos2_consensus=float(((U.T @ c1) ** 2).sum()),
                                         full_flip=cf.loc[k, "full_flip"], train_acc=cf.loc[k, "train_acc"], init_capture=cf.loc[k, "init_capture"]))
d = pd.DataFrame(rows).sort_values(["k", "cos2_consensus"]); print(d.round(3).to_string(index=False))
for k in (1, 2):
    x = d[d.k == k]; print(k, "Spearman(cos2_consensus, full_flip)", spearmanr(x.cos2_consensus, x.full_flip), "n", len(x))
x = d[d.k <= 2]; print("k<=2", spearmanr(x.cos2_consensus, x.full_flip), len(x))
