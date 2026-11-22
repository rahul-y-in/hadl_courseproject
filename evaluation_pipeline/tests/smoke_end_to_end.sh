#!/usr/bin/env bash
# End-to-end smoke test of steps 01-06 on CPU with a tiny, randomly initialised Qwen3 model
# (same tokenizer and chat template as Qwen3-1.7B). Checks that every code path runs;
# the scores are meaningless. Takes ~3-5 minutes on a laptop CPU. Real results are untouched.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WORK="${SMOKE_DIR:-$(mktemp -d)}"
export HADL_OUTPUT_ROOT="$WORK/outputs"
PY="${PYTHON:-python3}"
cd "$HERE"
echo "Smoke-test outputs in $WORK"

$PY tests/make_tiny_model.py "$WORK/tiny_qwen3"
$PY scripts/01_build_pool.py --random_pool_size 2000
$PY scripts/02_select_subsets.py --budget_x_real 0.05 --seeds 1 2 --sweep
$PY scripts/03_train.py --model_id "$WORK/tiny_qwen3" --methods all --reference_runs real_only --seeds 1 \
    --max_train_samples 24 --max_steps 3 --batch_size 2 --grad_accum 2 --max_seq_len 512 --dtype fp32
$PY scripts/04_evaluate.py --model_id "$WORK/tiny_qwen3" --max_eval_samples 6 --max_new_tokens 8 \
    --batch_size 4 --max_input_tokens 1024 --n_boot 200 --dtype fp32
$PY scripts/05_aggregate.py --n_boot 200
$PY scripts/06_make_plots.py
test -f "$HADL_OUTPUT_ROOT/results/results_table.csv"
test -f "$HADL_OUTPUT_ROOT/figures/qa_scores_test.png"
echo "SMOKE TEST PASSED"
