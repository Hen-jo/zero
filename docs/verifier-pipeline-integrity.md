# Verifier pipeline integrity repair

## Baseline and scope

This repair starts from remote `revthink-backward-verifier` commit
`7001070b756f2b98d4bebf33987b05d7c53fe739` (checked 2026-10-06). It does not
establish which version is running on the original training server. No training,
model inference, model download, remote push, or manuscript-result update was
performed for this change.

The restored verifier learns **normalized answer exact-match agreement with the
proposer's answer** (`answer_em_proxy`). Neither its supervision nor a YES verdict
establishes factual truth, evidence-chain validity, hop necessity, or causal
learning utility. A backward-search prompt is not an evidence intervention.

## Contracts and intentional behavior

1. Generation writes stable content `item_id`, `source_id`, `question`, and
   `reference_answers`, with original source text and generation provenance in
   metadata. The old 50-word question limit remains the default for reproducibility;
   `+data.question_word_limit=0` explicitly preserves the full question. Both the
   original and solver-facing question are retained. Empty questions/answers and
   duplicate generated IDs now fail rather than yielding ambiguous training data.
   Partition boundaries now retain remainder rows instead of dropping them.
2. Solver export preserves **all** sampled candidates. `rollout_uid` and
   `rollout_index` travel with candidates through token-length balancing, so neither
   adjacency nor equal row count is used to infer question identity. Each record
   contains item/source IDs, candidate ID, response, training reward, and generation
   step. `responses` contains only the final assistant message; the full decoded
   trace is preserved separately, so retrieved documents containing answer tags
   cannot masquerade as solver answers. Training reward remains distinct from EM.
3. Exports remain suffixes of on-policy training by default (last ten steps),
   potentially spanning several policy states. They are **not** final-checkpoint
   full-dataset evaluation. The builder requires full ID coverage unless partial
   coverage is explicitly accepted, and reports coverage in its manifest. Repeated
   items at different steps are legitimate distinct candidate samples.
4. The builder joins exclusively by stable IDs, checks content/provenance, and
   rejects duplicates, unknown candidates, and missing data. It does not recover
   legacy one-column `responses` exports through positional matching. Recover true
   mappings from trusted logs or create new ID-bearing artifacts; equal row counts
   cannot repair lost identity.
5. Train/validation splits keep the same source-text hash together. This prevents
   exact source-excerpt leakage among the provided records, not all possible
   document/semantic overlap: different excerpts from the same original document
   may need externally supplied document-level grouping in a future data audit.
   Validation remains synthetic answer-proxy evaluation, not an independent QA
   benchmark or chain-verification result. These sources are held out only from
   verifier optimization: the solver already trained on them and the challenger
   warm start may have seen them. This is not pipeline-level unseen-source
   generalization. Malformed/empty solver answers stop the build rather than being
   silently dropped or relabeled as negative examples.
6. Verifier verdict parsing is strict: malformed/missing/duplicate answer tags do
   not earn the reward for NO. Batch scoring uses structured message roles and
   requires a final assistant verdict, excluding answer tags in tool documents.
   Labels carry an explicit supervision source.
7. Proposer feedback is experimental and disabled by default. Existing iteration-2
   growth reward and iteration-3 difficulty-only configuration are retained; neither
   launcher is silently switched to another algorithm. Enabling verifier feedback
   adds compute and changes the objective, so record its resolved configuration as
   a separate experiment.

## Existing artifacts and checkpoints

The new builder intentionally refuses positional legacy exports and incompatible
QA validation labels. Verifier launchers use the builder's held-out verifier data.
Warm starts must name an actual Hugging Face model directory, not an unconverted
FSDP `global_step_*` checkpoint root. Conversion remains a separate explicit step
using the repository's supported tools; this patch does not run it.

No existing experimental artifact is rewritten by this development task. CPU
fixtures use temporary directories. Before a real run, audit the actual generated
Parquet files, coverage/class balance, tokenizer/prompt limits, retrieval access,
checkpoint conversion, resolved configuration, and framework versions.

## CPU checks

Run the dependency-free contract tests from the repository root:

```bash
python -m unittest discover -s tests -p 'test_*.py' -v
```

Parquet CLI integration additionally needs pandas and pyarrow; these tests must use
small synthetic fixtures, with no model downloads, model endpoints, GPUs, or
retrieval services. Source compilation and shell syntax checks do not substitute
for distributed Ray/FSDP/SGLang training verification. Full distributed training
and actual-server compatibility have not been exercised by this patch.

## Build and preflight (no training)

The standalone CPU builder supports these explicit paths:

```bash
python -m scripts.data_process.verifier_data \
  --gen_data_path /path/to/id-bearing-generated.parquet \
  --solver_pred_path /path/to/all-candidate-solver.parquet \
  --local_dir /path/to/new-verifier-data \
  --allow_subset
python -m scripts.verifier_preflight data \
  --train /path/to/new-verifier-data/train.parquet \
  --val /path/to/new-verifier-data/val.parquet
```

Omit `--allow_subset` to require full item coverage. Existing output artifacts are
protected unless `--overwrite` is explicitly provided. `dataset_manifest.json`
records candidate counts, item coverage, source counts, split seed, and class
balance. Inspect this manifest before committing GPU resources.

For a launcher's complete offline artifact checks and dataset build:

```bash
VERIFIER_INIT_MODEL=/path/to/converted/hf-model \
GEN_DATA_PATH=/path/to/id-bearing-generated.parquet \
SOLVER_PRED_PATH=/path/to/all-candidate-solver.parquet \
VERIFIER_DATA_DIR=/path/to/new-verifier-data \
VERIFIER_ALLOW_PARTIAL_COVERAGE=1 VERIFIER_PREFLIGHT_ONLY=1 \
bash iter1_verifier.sh 4321
```

The launcher also requires readable retrieval index/corpus and a local retriever
model layout; override `NFS_INDEX`, `NFS_CORPUS`, and `RETRIEVER_MODEL` as needed.
`BASE_MODEL` controls artifact names, while `VERIFIER_INIT_MODEL` independently
selects the warm start. `OUTPUT_DIR` controls the experiment root. The launcher
never guesses a conversion result, kills unrelated port owners, or starts the old
unused external verifier-reward SGLang server. It only cleans up retrieval that it
started itself. GPU training starts only when `VERIFIER_PREFLIGHT_ONLY` is unset
or zero; that operation was not run as part of this repair.

## Explicit experimental feedback

In an independently authorized run, optional proposer reward kwargs are:

```text
+custom_reward_function.reward_kwargs.verifier_feedback_enabled=true
+custom_reward_function.reward_kwargs.verifier_feedback_coefficient=0.1
+custom_reward_function.reward_kwargs.verifier_base_url=http://127.0.0.1:8003
+custom_reward_function.reward_kwargs.verifier_model_name=YOUR_VERIFIER_MODEL
+custom_reward_function.reward_kwargs.verifier_tokenizer_compatible=true
+custom_reward_function.reward_kwargs.verifier_prompt_length=1536
```

The coefficient above is illustrative, not experimentally selected. The endpoint
must use the exact compatible tokenizer/vocabulary expected by the caller; the
compatibility flag acknowledges this requirement and is not a remote fingerprint
check. Feedback uses one deterministic, tool-aware pass on each format-valid
proposed question and adds `coefficient * 1[verdict == YES]`. NO and malformed
verdicts both add zero, with separate validity diagnostics. This is an additive
answer-validity proxy bonus, not a chain-necessity objective. Transport/model
errors and response-count mismatches fail the run instead of silently replacing
its configured reward.

## Validation record (2026-10-06)

- 69 CPU tests and 118 parametrized subtests passed with pytest.
- Real synthetic Parquet round-trips cover exporter → strict join → dataset
  builder → held-out-data preflight. Guarded offline checks exercise all three
  verifier launchers, including partial coverage and paths with spaces.
- Role-grounding tests reject tool/user answer-tag spoofing, final-tool traces,
  unfinished final assistant turns, and duplicate verdicts.
- Mocked rollout tests verify disabled baseline parity and both enabled reward
  formulas without contacting an endpoint.
- Changed Python sources compile; all three verifier scripts pass `bash -n`;
  `git diff --check` passes.
- Ray/FSDP/SGLang distributed training, actual model weights/tokenizers, retrieval
  quality, GPU memory use, actual-server artifacts, and benchmark gains are
  unverified. No model/server/training process was started during these checks.
