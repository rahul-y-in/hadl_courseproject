# Evaluation pipeline: integration, metrics and plots

**Owner:** Rahul Yadav (Person 6, Evaluation & Reporting). This covers pipeline integration, evaluation metrics and plotting results.

This folder connects everyone's work into one reproducible pipeline and adds the evaluation and reporting layer:

```
Pratyush            Aditya Jain          Vanshika              Yatika              Aditya Sahu
real/synthetic  ->  low-rank scores  ->  covariance-matching -> time proxy      ->  Qwen3-1.7B + LoRA r=8
data splits         (LR pool)            ranking               c_i per example      training recipe
        \_______________________________________|______________________|__________________/
                                                v
                           evaluation_pipeline/ (this folder)
   01 costed pool -> 02 budgeted selection (5 methods) -> 03 train -> 04 evaluate -> 05 tables -> 06 figures
```

Everything except steps 3 and 4 runs on a laptop CPU in about 2 minutes. Steps 3 and 4 need one GPU (Colab A100/L4/T4 or Kaggle 2×T4).

---

## 1. Quick start

### On a cloud GPU (recommended)

Open `notebooks/run_pipeline_colab.ipynb` in Colab or Kaggle and run all cells.

> **Before running:** the notebook clones `main`. While this work is still on its feature branch, set `BRANCH` in the first cell to that branch.

The notebook runs this single command:

```bash
PROFILE=1 SEEDS="42" REFERENCE="real_only" bash evaluation_pipeline/run_pipeline.sh
```

| Variable | Default | Meaning |
|---|---|---|
| `PROFILE=1` | off | Re-profile the LoRA step time **on this GPU** and fit a new time proxy. Recommended, because the committed proxy was profiled on Apple MPS. |
| `BUDGET_X_REAL` | `1.0` | Synthetic training-time budget, as a multiple of the proxy time of one epoch over the 886 real training rows. |
| `BUDGET_SEC` | — | Absolute budget in proxy seconds (overrides `BUDGET_X_REAL`). |
| `SEEDS` | `42` | Seeds for training and for the random baseline. Use `"42 43 44"` for error bars. |
| `METHODS` | the three headline methods | Set `all` to add the `low_rank_only` and `cov_match_only` ablations. |
| `REFERENCE` | — | `real_only` also fine-tunes on the real data alone. |
| `NPROC` | `1` | Use `2` on Kaggle 2×T4 (torchrun DDP). |
| `DTYPE` | `auto` | `auto` picks bf16 on A100/L4 and fp16 on T4. Use `fp32` if fp16 gives NaN losses. |
| `EVAL_BS` | `8` | Evaluation batch size. Use 4 on a T4. |
| `STEPS` | `0 1 2 3 4 5 6` | For example, `STEPS="5 6"` only re-aggregates and re-plots. |

Finished runs and evaluations are skipped automatically. After a Colab disconnect, re-running the same command resumes where it stopped.

### Step by step

```bash
cd evaluation_pipeline
pip install -r requirements.txt
python scripts/01_build_pool.py          # CPU, ~90 s:  integrate LR + CM + time proxy
python scripts/02_select_subsets.py --sweep   # CPU, ~2 s: one subset per method
python scripts/03_train.py --reference_runs real_only   # GPU
python scripts/04_evaluate.py            # GPU: base model + every adapter, val + test
python scripts/05_aggregate.py           # CPU: tables + significance tests + proxy error
python scripts/06_make_plots.py          # CPU: all figures (PNG + PDF)
```

Every script has `--help`.

---

## 2. What is compared

All methods get **the same synthetic training-time budget B** (proxy-predicted seconds per epoch). They are trained with the same recipe on **real train (886 rows) + the selected synthetic subset**, and evaluated on the real IndicQA Odia **validation (195)** and **test (176)** splits.

| Run | Selection rule | Role |
|---|---|---|
| `base_zero_shot` | No fine-tuning | Reference |
| `real_only` | Real data only | Reference (optional) |
| `random` | Uniformly random over the full 185k synthetic train pool; prefix fill | Baseline 1 |
| `quality_only` | Composite usefulness `u_i` (LR + CM), highest first, **cost ignored**; prefix fill | Baseline 2: "the two methods combined" |
| `proposed_knapsack` | **max Σu_i subject to Σc_i ≤ B** (0/1 knapsack), with `c_i` from the time proxy | **Our method** |
| `low_rank_only` | Low-rank score only | Ablation |
| `cov_match_only` | Greedy covariance-matching order only | Ablation |

**How the scores are integrated** (`hadl_eval/pool.py`):

- `lr_score` is the low-rank stage's `sampling_probability`, which is proportional to `max(pred_loss,0) + λ·residual²`. `lr_pct` is its percentile rank.
- `cm_score = 1 − cm_rank / 7088`, from the greedy covariance-matching order. Rows that are not ranked get 0.
- `u_i = α·lr_pct + (1−α)·cm_score`, with α = 0.5 (`--alpha` in step 1).
- `c_i` = Yatika's proxy, `T_step(B=4, L_i)/4`. Here `L_i` is the **exact tokenised training length** (prompt + answer, after truncation).
- `quality_only` and `proposed_knapsack` choose from the same 7,088 covariance-matched candidates. The only difference between them is whether the time cost is used, which isolates the contribution of the time proxy.

**Knapsack solver** (`hadl_eval/knapsack.py`): an exact DP on costs discretised into 2,000 units (rounded **up**, so the solution is always feasible), plus a greedy u/c pass. The better of the two is kept, and any remaining budget is filled greedily. The tests check it against brute force.

**Budget choice.** Absolute seconds depend on the GPU the proxy was profiled on. The default budget is therefore `1.0 × (proxy time of the real train split)`. With the committed MPS proxy this is 8,605 s, and the methods select 1.6k–3.4k examples. Re-profiling on the A100 changes the seconds but keeps the same meaning.

---

## 3. Metrics

### QA quality (`hadl_eval/metrics.py`), per example, taking the maximum over gold answers

| Metric | Definition |
|---|---|
| **Exact match (EM)** | Normalised strings are equal. Normalisation: NFC, remove zero-width characters, Odia/Devanagari digits → ASCII, remove punctuation including `।`, collapse spaces. |
| **Token F1** | SQuAD-style overlap of whitespace tokens. This is the main metric. |
| **Character F1** | Overlap of characters. It is robust to Odia suffixes (`ଭୁବନେଶ୍ୱର` vs `ଭୁବନେଶ୍ୱରରେ`). |
| **Answer contained** | The normalised gold answer appears inside the prediction. |

- **95% CIs:** bootstrap resampling **clustered by passage**, because questions about the same passage are correlated.
- **Significance:** a paired, passage-clustered bootstrap of each run against the random baseline on the same examples (difference, CI, two-sided p).

### Training-time proxy

| Level | What is compared |
|---|---|
| Run level | Proxy-predicted total vs measured wall-clock training time: MAPE, signed error, R², Spearman. Also a **leave-one-out calibrated MAPE**, which shows the proxy error once its hardware constant is fitted. |
| Micro-batch level | Every forward+backward is timed (CUDA-synchronised) together with its real `(B, padded L)`. This tests the functional form of the proxy, not only its total. |

### Efficiency

- Selection time per method
- Proxy-scoring time per example
- Training wall-clock time, throughput (tokens/s), peak GPU memory, padding fraction
- Generation time per example
- One-off stage costs for embeddings, low-rank scoring and covariance matching. Enter these in `configs/overhead_times.json`; they appear in the overhead plot.

---

## 4. Outputs

```
outputs/
  pool/            candidate_pool.parquet, random_pool.parquet, real_train.parquet, pool_report.json   (git-ignored, regenerable)
  selections/      budgets.json, budget_sweep.csv, b<B>/<run_id>.parquet, b<B>/selection_summary.json
  runs/<run_id>/   adapter/, training_metrics.json, log_history.json, step_timings.csv
  eval/<run_id>/   {validation,test}_predictions.jsonl, {validation,test}_metrics.json
  results/         results_table.{csv,md}, results_by_method.{csv,md}, significance_vs_random.{csv,md},
                   proxy_runs.csv, proxy_error_summary.json, overhead_components.csv, selection_summary.csv
  figures/         *.png (200 dpi) + *.pdf (vector, for the report)
```

Run ids look like `proposed_knapsack_b8605_s42`, meaning method, budget in seconds, and seed.

| Figure | Question it answers |
|---|---|
| `qa_scores_{split}` | Which method answers best? EM / F1 / char-F1 with 95% CIs. |
| `quality_vs_training_time_{split}` | Quality against the training time actually spent (Pareto view). |
| `token_f1_vs_budget_{split}` | How quality scales with the budget (needs ≥ 2 budgets). |
| `token_f1_by_answer_length_{split}` | Where the methods differ (short vs long answers). |
| `training_time_predicted_vs_actual` | How accurate the time proxy is per run. |
| `proxy_step_parity` | Proxy vs measured time per micro-batch, and where it is off. |
| `proxy_profile_fit` | The proxy curves on the profiling grid. |
| `training_loss_curves` | Training dynamics. |
| `selection_overview_b<B>` | What each method selects: count, Σu, length, diversity. |
| `selection_token_lengths_b<B>` | Length distribution of the selected examples. |
| `utility_vs_cost_b<B>` | Why the knapsack picks more examples (high u/c). |
| `budget_sweep_total_utility`, `budget_sweep_num_selected` | The selection objective across budgets (no GPU needed). |
| `budget_sweep_cov_shift_rel` | Covariance distance to the real data, when the embedding `.pt` files are present. |
| `pipeline_overhead` | Selection pipeline overhead plus training cost. |

**Already produced (CPU):** `outputs/selections/` and the selection and proxy figures in `outputs/figures/` are committed. At the default budget, the knapsack selects **3,433** examples with Σu = 1,902. The no-time-proxy baseline selects 1,630 examples with Σu = 1,223, and random selects 2,152 examples with Σu = 447.

---

## 5. Integration issues found and fixed

These were found while connecting the stages. The original files are left untouched, and the corrected logic lives in `hadl_eval/`.

1. **Placeholder scores.** `model_training/prepare_training_subsets.py` looks for `low_rank_score` / `cov_match_score` columns that no stage writes. It therefore fell back to random "synthetic_fallback" scores (see `subset_selection_summary.json`). This pipeline uses the real LR probabilities and the CM ranking.
2. **Empty passages.** The synthetic files call the passage `paragraph`, but `assign_time_costs.py` and `train_qwen_lora.py` read `context`. That silently gave an **empty passage**, so models were trained without context and costs were underestimated. All data now goes through `hadl_eval/data.py` with one schema.
3. **Truncation cut the question.** Training truncated `prompt_ids[:max_len]`, which cuts the *end* of the prompt: the question and the assistant header. Now only the passage is shortened, and for training the kept window contains the answer.
4. **512-token limit removed the answer.** Odia costs about 2 Qwen tokens per character. Real test passages have a median of ~1,300 tokens and a maximum of ~3,700. Evaluation now uses 4,096 input tokens, so no passage is cut. Training uses 2,048, which is the proxy's cap.
5. **Different train and eval prompts.** Training and evaluation used different system prompts. Both now use `hadl_eval/prompts.py`.
6. **Synthetic answer format.** Synthetic answers always end with `।`, while IndicQA answers are bare spans. The trailing punctuation is now stripped from targets.
7. **fp16 on T4.** fp16 training on T4 needs the LoRA weights in fp32 (otherwise "Attempting to unscale FP16 gradients"). The weights are now cast.
8. **Dual-GPU split.** `device_map="auto"` on 2×T4 splits the model across GPUs and distorts timings. Now there is one GPU per process, and DDP via torchrun.
9. **Evaluation size.** Evaluation previously used 10 validation examples with batch size 1. It now runs the full splits batched, with CIs.

The candidate pool was also checked for leakage. **0** of the 33,104 candidates share an 8-word shingle with a real validation or test passage, and 1 of the 30,000 random-pool rows did and was dropped.

---

## 6. Expected GPU time (rough estimates)

| Hardware | One training run (~4.3k rows, 1 epoch) | Evaluation per model (371 questions) | Default pipeline (3 methods + real_only + base) |
|---|---|---|---|
| A100 40/80 GB | ~15–30 min | ~3–6 min | ~2 h |
| L4 / single T4 | ~1–2 h | ~10–20 min | ~6–8 h. Use `BUDGET_X_REAL=0.5` or fewer methods. |

These are estimates; the real numbers are measured and reported by the pipeline itself.

Troubleshooting:

| Problem | Fix |
|---|---|
| OOM during training | `--max_seq_len 1536` or `--batch_size 2 --grad_accum 8`. Keep the same proxy batch size via `01_build_pool.py --batch_size`. |
| OOM during evaluation | `EVAL_BS=2`. The OOM handler also splits batches automatically. |
| NaN loss with fp16 | `DTYPE=fp32`. |
| No internet for the tokenizer | The Qwen3 tokenizer copy in `model_training/checkpoints/*/final_adapter` is used automatically. |

---

## 7. Code layout and tests

```
hadl_eval/   config.py (paths, hyper-parameters, colours) · data.py · prompts.py · text_utils.py
             time_proxy.py · pool.py · knapsack.py · selection.py · diagnostics.py
             modeling.py · training.py · inference.py · metrics.py · results.py · plotting.py
scripts/     01_build_pool · 02_select_subsets · 03_train · 04_evaluate · 05_aggregate · 06_make_plots
tests/       unit tests (pytest) + smoke_end_to_end.sh (all 6 steps on CPU with a tiny random Qwen3)
notebooks/   run_pipeline_colab.ipynb
configs/     overhead_times.json
```

```bash
python -m pytest -q tests                      # 46 unit tests, ~5 s
bash tests/smoke_end_to_end.sh                 # full pipeline on CPU with a tiny model, ~5 min
```

The smoke test runs training, evaluation, aggregation and plotting end to end without a GPU. Its scores are meaningless, because the tiny model is random. It exists to check that every code path executes. Set `HADL_OUTPUT_ROOT` to write outputs elsewhere, as the smoke test does.
