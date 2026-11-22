#!/usr/bin/env bash
# Full HADL evaluation pipeline. Run from anywhere:
#
#   bash evaluation_pipeline/run_pipeline.sh                 # everything, default settings
#   PROFILE=1 bash evaluation_pipeline/run_pipeline.sh       # first re-profile the time proxy on THIS GPU
#   SEEDS="42 43 44" BUDGET_X_REAL="0.5 1 2" METHODS=all REFERENCE="real_only" bash evaluation_pipeline/run_pipeline.sh
#   STEPS="5 6" bash evaluation_pipeline/run_pipeline.sh     # only aggregate + plot
#
# Environment variables (all optional):
#   STEPS          which steps to run (default "0 1 2 3 4 5 6"; step 0 runs only with PROFILE=1)
#   PROFILE=1      re-profile Qwen3-1.7B LoRA step time on the local GPU and use that proxy
#   PROXY_JSON     time-proxy json to use (default: performance_profiling/time_proxy_model.json)
#   BUDGET_X_REAL  synthetic budget(s) as multiples of the real-train proxy time (default "1.0")
#   BUDGET_SEC     absolute budget(s) in proxy seconds (overrides BUDGET_X_REAL)
#   SEEDS          training / random-selection seeds (default "42")
#   METHODS        methods to train: "random quality_only proposed_knapsack" (default) or "all"
#   REFERENCE      extra reference runs, e.g. "real_only"
#   DTYPE          auto | bf16 | fp16 | fp32 (default auto)
#   EPOCHS         training epochs (default 1)
#   EVAL_BS        evaluation batch size (default 8; lower it on a T4 if you hit OOM)
#   NPROC          GPUs for training via torchrun DDP (default 1; use 2 on a dual-T4 Kaggle box)
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/.." && pwd)"
PY="${PYTHON:-python}"
STEPS="${STEPS:-0 1 2 3 4 5 6}"
SEEDS="${SEEDS:-42}"
METHODS="${METHODS:-random quality_only proposed_knapsack}"
REFERENCE="${REFERENCE:-}"
DTYPE="${DTYPE:-auto}"
EPOCHS="${EPOCHS:-1}"
EVAL_BS="${EVAL_BS:-8}"
BUDGET_X_REAL="${BUDGET_X_REAL:-1.0}"
USER_PROXY="${PROXY_JSON:-}"
PROXY_JSON="${PROXY_JSON:-$ROOT/performance_profiling/time_proxy_model.json}"
NPROC="${NPROC:-1}"
OUT="${HADL_OUTPUT_ROOT:-$HERE/outputs}"

has_step() { [[ " $STEPS " == *" $1 "* ]]; }
cd "$HERE"

if has_step 0 && [[ "${PROFILE:-0}" == "1" ]]; then
  echo "=== Step 0: profile LoRA step time on this GPU and fit the time proxy"
  PROF_DIR="$OUT/profiling"
  mkdir -p "$PROF_DIR"
  (cd "$PROF_DIR" && "$PY" "$ROOT/performance_profiling/profile_hardware.py" \
                  && "$PY" "$ROOT/performance_profiling/fit_time_proxy.py")
  PROXY_JSON="$PROF_DIR/time_proxy_model.json"
  export HADL_PROFILING_CSV="$PROF_DIR/profiling_results.csv"
  echo "Using freshly fitted proxy: $PROXY_JSON"
elif [[ -z "$USER_PROXY" && -f "$OUT/profiling/time_proxy_model.json" ]]; then
  PROXY_JSON="$OUT/profiling/time_proxy_model.json"
  export HADL_PROFILING_CSV="$OUT/profiling/profiling_results.csv"
  echo "Re-using proxy profiled earlier on this machine: $PROXY_JSON"
fi

if has_step 1; then
  echo "=== Step 1: build the costed candidate pool"
  "$PY" scripts/01_build_pool.py --proxy_json "$PROXY_JSON"
fi

if has_step 2; then
  echo "=== Step 2: select subsets"
  if [[ -n "${BUDGET_SEC:-}" ]]; then
    "$PY" scripts/02_select_subsets.py --budget_sec $BUDGET_SEC --seeds $SEEDS --sweep
  else
    "$PY" scripts/02_select_subsets.py --budget_x_real $BUDGET_X_REAL --seeds $SEEDS --sweep
  fi
fi

if has_step 3; then
  echo "=== Step 3: fine-tune Qwen3-1.7B + LoRA"
  REF_ARGS=()
  [[ -n "$REFERENCE" ]] && REF_ARGS=(--reference_runs $REFERENCE)
  if [[ "$NPROC" -gt 1 ]]; then LAUNCH=(torchrun --nproc_per_node "$NPROC"); else LAUNCH=("$PY"); fi
  "${LAUNCH[@]}" scripts/03_train.py --methods $METHODS ${REF_ARGS[@]+"${REF_ARGS[@]}"} --seeds $SEEDS --dtype "$DTYPE" \
        --epochs "$EPOCHS" --proxy_json "$PROXY_JSON"
fi

if has_step 4; then
  echo "=== Step 4: evaluate on the real validation + test splits"
  "$PY" scripts/04_evaluate.py --splits validation test --dtype "$DTYPE" --batch_size "$EVAL_BS"
fi

if has_step 5; then
  echo "=== Step 5: aggregate results"
  "$PY" scripts/05_aggregate.py
fi

if has_step 6; then
  echo "=== Step 6: plots"
  "$PY" scripts/06_make_plots.py --proxy_json "$PROXY_JSON"
fi

echo "Done. Tables: $OUT/results   Figures: $OUT/figures"
