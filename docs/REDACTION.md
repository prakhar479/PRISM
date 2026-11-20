# Redaction for anonymous review (2026-11-20)

This snapshot was anonymized for review. No measured value, configuration, prompt or analysis result was changed.

## What was replaced, in every text file
| category | where it occurred | replaced by |
|---|---|---|
| author names | the preregistration addendum (one line: who approved the template packs), `data/lang/signoff.json` (`by` field), the lab notebook | "an author (name redacted for review)" |
| user names, e-mail addresses, home directories | run manifests and logs, queue state, analysis scripts and reports | `user`, `<email>`, `/home/user`, `<repo>`, `<workspace>` |
| machine and host names | run manifests and logs (`hostname`) | `node-A` … `node-D` for the four GPU server nodes, `laptop`, `dev-machine`, `login-node` |
| scratch workspace paths | analysis scripts and reports | `.layout` (see `analysis/*/scripts/layout_shim.py`), `<tmp>` |
| repository URL, institution | documentation | `<repository-url>`, `institution` |

## Golden fixtures
The golden fixtures (`golden_parts/`) were re-packed:
- the tar archive's owner fields were set to `user`;
- one JSON member's host name was redacted;
- member contents and order are otherwise unchanged;
- the archive was re-hashed (`golden_parts/R2.tar.sha256`) and split into parts under 45 MB.

## Effects on integrity checks
- **Preregistration addendum:** `PREREGISTRATION_ADDENDUM.md` differs from the frozen original in that one line only.
  `tests/harness/test_prereg_frozen.py` pins the redacted file's hash and records the original's
  (`4ba48a08…f4aaa`). `PREREGISTRATION.md` is unchanged.
- **Template sign-offs:** they are bound to each pack's content hash, not to the name, so validation is unaffected.
- **Git commit ids in run manifests** refer to the development history, which is not part of this single-commit snapshot.
- **Shipping checksums:** the files a run shipped are checksummed in bundle manifests at shipping time. Redacted logs and
  manifests no longer match those pre-redaction checksums. Every data file under `results/runs/` and `tables/` (parquet, npz, csv) is byte-identical to the original (checked).

## Verification
A byte-level scan over every file in this snapshot, text and binary, finds none of the redacted strings.
