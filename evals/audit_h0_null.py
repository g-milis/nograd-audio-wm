"""Re-score compact unwatermarked token fixtures with the released h=0 detector.

Run from the repository root: python -m evals.audit_h0_null --check-reference
No generator, codec checkpoint, or audio reconstruction is used.
"""
import argparse
import json
import math
from pathlib import Path

import torch
from evals.main_wm import load_clustering_maps, compute_watermark_scores, get_binomial_pval

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_FIXTURE = ROOT / "tests/fixtures/moshi_h0_unwatermarked_unique_tokens.json"


def audit(fixture_path, key_seed=0, threshold=0.01):
    fixture = json.loads(Path(fixture_path).read_text())
    config = fixture["detection"]
    if fixture["format"] != "moshi-h0-null-unique-tokens-v1" or config["h"] != 0:
        raise ValueError("This compact representation supports h=0 only")
    if not 0 < threshold < 1:
        raise ValueError("threshold must be between zero and one")
    selection = json.loads((ROOT / config["cluster_selection"]).read_text())
    loaded = load_clustering_maps(str(ROOT / config["cluster_asset"]),
                                  select_map=selection, device="cpu")
    maps = {}
    for stream in config["streams"]:
        candidates = [value for method in loaded[stream - 1].values() for value in method.values()]
        if len(candidates) != 1:
            raise ValueError(f"Expected one selected clustering for stream {stream}")
        maps[stream] = candidates[0][1]
    empty_context = torch.zeros((1, 0), dtype=torch.long)
    rows = []
    for sample in fixture["samples"]:
        if len(sample["unique_token_ids"]) != len(config["streams"]):
            raise ValueError("Fixture stream count differs from the detection configuration")
        for detector in ("base", "clustered"):
            green_counts, scored_counts = [], []
            for stream, values in zip(config["streams"], sample["unique_token_ids"]):
                if values != sorted(set(values)) or any(v < 0 or v >= config["audio_vocab_size"] for v in values):
                    raise ValueError("Expected sorted unique raw-token IDs in the codec vocabulary")
                tokens = torch.tensor(values, dtype=torch.long)
                green, scored = compute_watermark_scores(
                    tokens, empty_context, config["audio_vocab_size"], config["gamma"],
                    key_seed, device="cpu", clustering_map=maps[stream] if detector == "clustered" else None)
                green_counts.append(int((green & scored).sum()))
                scored_counts.append(int(scored.sum()))
            p_value = float(get_binomial_pval(sum(green_counts), sum(scored_counts), config["gamma"]))
            if not math.isfinite(p_value):
                raise ValueError("The released tail calculation returned a nonfinite value")
            reference_match = None
            if key_seed == config["key_seed"]:
                ref = sample["reference_scores"][detector]
                reference_match = (green_counts == ref["green_counts"]
                                   and scored_counts == ref["scored_counts"]
                                   and math.isclose(p_value, ref["p_value"], rel_tol=1e-12, abs_tol=0.0))
            rows.append(dict(prompt=sample["prompt"], detector=detector, key_seed=key_seed,
                             green_counts=green_counts, scored_counts=scored_counts,
                             p_value=p_value, accepted=p_value <= threshold,
                             reference_match=reference_match))
    count = len(fixture["samples"])
    if not count:
        raise ValueError("Empty fixture")
    return {"h": 0, "key_seed": key_seed, "nominal_threshold": threshold, "null_audio_count": count,
            "counts": {name: {"false_positives": sum(row["accepted"] for row in rows if row["detector"] == name),
                              "denominator": count} for name in ("base", "clustered")},
            "reference_matches": sum(row["reference_match"] is True for row in rows),
            "reference_comparisons": sum(row["reference_match"] is not None for row in rows),
            "rows": rows}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixture", type=Path, default=DEFAULT_FIXTURE)
    parser.add_argument("--key-seed", type=int, default=0)
    parser.add_argument("--threshold", type=float, default=0.01)
    parser.add_argument("--check-reference", action="store_true",
                        help="Check against the captured release-version scores; not a requirement for future corrected detectors")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = audit(args.fixture, args.key_seed, args.threshold)
    text = json.dumps(result, indent=2, allow_nan=False) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text)
    print(text, end="")
    if args.check_reference and (result["reference_comparisons"] != 2 * result["null_audio_count"]
                                or result["reference_matches"] != result["reference_comparisons"]):
        raise SystemExit("Scores differ from the captured release snapshot (or a different key was requested)")


if __name__ == "__main__":
    main()
