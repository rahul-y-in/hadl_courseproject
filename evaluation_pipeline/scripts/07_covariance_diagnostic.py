"""Step 7 (optional, CPU only, run AFTER the Kaggle/Colab run) - covariance-shift diagnostic.

How close is each method's selected synthetic subset to the covariance of the real IndicQA
train data? This is the quantity covariance matching minimises, so it shows whether
covariance matching really shaped each subset and whether the time-aware knapsack kept or
undid that. No GPU and no re-training needed: it only reads the subsets the run saved.

Needs the two git-ignored embedding files (ask Aditya Jain / Vanshika):
    embedding_generation/odia_embeddings.pt             (~665 MB, 216,394 x 768, all synthetic rows)
    embedding_generation/odia_real_train_embeddings.pt  (~3 MB,   886 x 768, real train rows;
                                                         written by covariance_matching.py)

Usage after the Kaggle run (see README section "After the GPU run: covariance diagnostic"):
    unzip hadl_results.zip -d kaggle_run           # -> kaggle_run/outputs/...
    python evaluation_pipeline/scripts/07_covariance_diagnostic.py --outputs kaggle_run/outputs

Writes into <outputs>/results/ and <outputs>/figures/:
    covariance_diagnostic.csv / .md       one row per subset + reference rows
    covariance_shift_b<B>.png / .pdf      bar chart per budget (lower = closer to real data)
"""
import argparse
import json
import time
from pathlib import Path

import _bootstrap  # noqa: F401
import numpy as np
import pandas as pd

from hadl_eval import config, plotting
from hadl_eval.diagnostics import CovarianceDiagnostic
from hadl_eval.pool import find_cm_ranking
from hadl_eval.results import to_markdown


def budget_from_dir(name: str) -> float:
    """'b5028' -> 5028.0, 'b8605p5' -> 8605.5 (inverse of config.budget_tag)."""
    try:
        return float(name[1:].replace("p", "."))
    except ValueError:
        return float("nan")


def subset_files(selection_dir: Path, budgets: list[float] | None) -> list[Path]:
    files = sorted(p for p in selection_dir.glob("b*/*.parquet"))
    if budgets is not None:
        tags = {f"b{config.budget_tag(b)}" for b in budgets}
        files = [p for p in files if p.parent.name in tags]
    return files


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--outputs", default=str(config.OUTPUT_ROOT),
                    help="the run's outputs folder (contains selections/); results are written next to it")
    ap.add_argument("--synth_emb", default=str(config.SYNTH_EMBEDDINGS_PT))
    ap.add_argument("--real_emb", default=str(config.REAL_TRAIN_EMBEDDINGS_PT))
    ap.add_argument("--pca_dim", type=int, default=64, help="PCA dimension (64 = covariance_matching.py default)")
    ap.add_argument("--all_budgets", action="store_true",
                    help="use every budget folder found, not only those listed in selections/budgets.json")
    args = ap.parse_args()

    out = Path(args.outputs).resolve()
    sel_dir = out / "selections"
    if not sel_dir.is_dir():
        raise SystemExit(f"{sel_dir} not found - point --outputs at the run's 'outputs' folder")
    for p in (args.synth_emb, args.real_emb):
        if not Path(p).is_file():
            raise SystemExit(f"Embedding file missing: {p}\n"
                             "Ask Aditya Jain (odia_embeddings.pt) / Vanshika (odia_real_train_embeddings.pt) "
                             "and put them in embedding_generation/, or pass --synth_emb / --real_emb.")

    budgets = None
    budgets_json = sel_dir / config.BUDGETS_JSON_NAME
    if budgets_json.is_file() and not args.all_budgets:
        budgets = [float(b) for b in json.loads(budgets_json.read_text())["budgets_sec"]]
    files = subset_files(sel_dir, budgets)
    if not files:
        raise SystemExit(f"No subset parquet files under {sel_dir} for budgets {budgets}")
    print(f"{len(files)} subset file(s) for budget(s) {budgets or 'all'}")

    t0 = time.perf_counter()
    diag = CovarianceDiagnostic(args.synth_emb, args.real_emb, pca_dim=args.pca_dim)
    print(f"Loaded embeddings: synthetic {diag.synth.shape}, real n={diag.n_real} ({time.perf_counter() - t0:.0f}s)")
    n_synth_rows = len(pd.read_parquet(config.SYNTH_FILTERED, columns=["question"]))
    if len(diag.synth) != n_synth_rows:
        raise SystemExit(f"{args.synth_emb} has {len(diag.synth)} rows but or_synthetic_filtered.parquet has "
                         f"{n_synth_rows}; _source_row_idx would not line up")

    rows = []
    seen = {}
    for p in files:
        df = pd.read_parquet(p, columns=["_source_row_idx"])
        key = (p.parent.name, tuple(np.sort(df["_source_row_idx"].values)))
        d = seen.get(key) or diag.distances(df["_source_row_idx"].values)
        seen[key] = d
        run_id = p.stem
        parts = run_id.rsplit("_s", 1)
        rows.append({"kind": "subset", "run_id": run_id, "method": config.method_of_run(run_id),
                     "budget_sec": budget_from_dir(p.parent.name),
                     "seed": int(parts[1]) if len(parts) == 2 and parts[1].isdigit() else np.nan, **d})

    # reference rows: what the selection stages start from
    lr = pd.read_parquet(config.LR_DRAWS, columns=["_source_row_idx"])["_source_row_idx"].unique()
    cm_path = find_cm_ranking()
    cm = pd.read_parquet(cm_path, columns=["_source_row_idx", "cm_rank"]).sort_values("cm_rank")
    refs = {"Whole low-rank pool": lr, f"All {len(cm):,} covariance-matched candidates": cm["_source_row_idx"].values}
    for n in (1772, 3544):
        if n < len(cm):
            refs[f"Covariance-matching top {n:,}"] = cm["_source_row_idx"].values[:n]
    for name, idx in refs.items():
        rows.append({"kind": "reference", "run_id": name, "method": "reference", "budget_sec": np.nan,
                     "seed": np.nan, **diag.distances(idx)})

    table = pd.DataFrame(rows)
    (out / "results").mkdir(parents=True, exist_ok=True)
    table.to_csv(out / "results" / "covariance_diagnostic.csv", index=False)
    show = table[["kind", "run_id", "n", "rel_dist_pca", "rel_dist_full"]]
    (out / "results" / "covariance_diagnostic.md").write_text(
        f"Relative covariance distance to the real train data (lower = closer). "
        f"PCA space k={args.pca_dim}; full = {diag.synth.shape[1]}-dim.\n\n" + to_markdown(show, floatfmt="{:.3f}"))
    print(to_markdown(show, floatfmt="{:.3f}"))

    written = plotting.plot_covariance_diagnostic(table, out / "figures")
    print(f"Wrote {out / 'results' / 'covariance_diagnostic.csv'} and {len([w for w in written if w.suffix == '.png'])} figure(s) "
          f"to {out / 'figures'}")


if __name__ == "__main__":
    main()
