# A small h=0 null-control reproducer

This diagnostic asks whether the released binomial p-values at a fixed key can be interpreted as false-positive rates. It does not change the watermark, clustering, counting rule, or tail calculation.

With `h=0`, `wm_seed=0`, and `gamma=0.25`, the released clustered scorer accepts all 10 unwatermarked continuations in the attached fixture at `p <= 0.01`. The unclustered Base scorer accepts 1/10 on exactly the same received tokens. These are empirical counts on a small fixed batch, not estimates of a universal false-positive rate.

## Recompute on CPU

From an installed repository environment:

```bash
CUDA_VISIBLE_DEVICES="" python -m evals.audit_h0_null --check-reference
```

The command directly calls `load_clustering_maps`, `compute_watermark_scores`, and `get_binomial_pval` from `evals.main_wm`. It reads the existing selected Mimi cluster asset and does not load a generator/codec checkpoint or reconstruct audio. The optional `SyncPattern` import is deferred to the existing `wm_sync` branch, so the diagnostic does not require AudioSeal when synchronization is disabled.

Expected counts for the captured release snapshot:

```text
base:       1 / 10 false positives
clustered: 10 / 10 false positives
reference_matches: 20 / 20
```

`--check-reference` checks the recorded counts and p-values. It is a reproduction check for this snapshot, not an assertion that a future corrected detector should retain these counts. Omit it when investigating a different key, for example `--key-seed 2070661460`.

## Fixture provenance

`tests/fixtures/moshi_h0_unwatermarked_unique_tokens.json` contains the unique raw-token IDs from the first four **memory-reencoded** Mimi streams of 10 unwatermarked Moshi outputs. The input files are the repository's `data/audio_prompts/prompt_00000.wav` through `prompt_00009.wav`.

Generation used `kyutai/moshiko-pytorch-bf16`, original Mimi, full prompts, batch size 1, global seed 42424242, temperature 1, 200 output frames, `wm_method=none`, and `wm_delta=0`. The outputs were re-encoded before WAV serialization, following the original evaluator's observation boundary. The first four received streams were then scored with both released detectors. The fixture records Python, Torch, NumPy, and SciPy versions.

Only sorted unique token IDs are published. At h=0, the partition is constant and the scorer already counts each raw-token ID once per stream. Removing order and repetitions preserves its green counts, trial counts, and p-values. This reduction is **not valid for h>0**. Audio, generated text, full token sequences, and local paths are not included.

Collection used local instrumentation to save tensors, results, and configuration. The local evaluator also deferred the optional sync import and used `idx` instead of `0` for the source-token batch row, which is identical for B1. The research checkout had an optional multi-bit sampling hook, but it was not configured in these runs; the ordinary `none` and Maryland h0 branches, original cluster assets, and Mimi encoding were used. The CPU reproducer in this PR runs directly on the upstream scorer rather than those research adapters.

## A fresh null-control command to try

The evaluator can be asked to score unwatermarked outputs with the clustered detector directly:

```bash
python -m evals.main_wm \
  --output_dir outputs/h0_null_control \
  --audio_dir data/audio_prompts \
  --nsamples 10 --batch_size 1 --steps 200 --temperature 1.0 \
  --seed 42424242 --wm_seed 0 --wm_ngram 0 \
  --wm_streams 1 2 3 4 --wm_gamma 0.25 \
  --wm_method none --wm_delta 0 --wm_clustering true \
  --eval_aug false --save_audio 10
```

Supply the usual local checkpoint arguments if needed. This is a suggested fresh generation check; it was not rerun for this PR. The attached batch was generated with clustering disabled and subsequently scored with the existing cluster maps, as described above. The committed CPU fixture is the tested reproducer.

## Questions for the authors

1. Is this the intended way to construct a matched, unwatermarked null control for h=0? Is an additional key restriction, centering step, or calibration procedure required?
2. How should the binomial null account for unequal cluster occupancy and several distinct raw-token IDs sharing the same green-cluster indicator? Could that explain the fixed-key result here?
3. Is this related to the issue mentioned in the README? Which code/configuration should be used as the reproduction reference?
4. Could you clarify the keys/seeds and input split used in Figures 3–5, and whether the theoretical-FPR curves were also checked against generated unwatermarked controls?

There are only 10 released prompts here, not the paper's full 500-example dataset. This example does not establish behavior for all keys or higher-order contexts, and it does not by itself identify the cause of the issue mentioned in the README.
