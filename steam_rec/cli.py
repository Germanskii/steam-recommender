"""Evaluate a demo or locally supplied Steam-200k file."""

import argparse
import json
from pathlib import Path
from steam_rec.data import demo_matrix, load_steam_csv
from steam_rec.model import predict, recommend
from steam_rec.evaluate import evaluate


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", type=Path)
    parser.add_argument("--output", type=Path, default=Path("results"))
    parser.add_argument(
        "--similarity", choices=["cosine", "jaccard", "pearson", "mixed"], default="cosine"
    )
    args = parser.parse_args()
    frame = load_steam_csv(args.csv) if args.csv else demo_matrix()
    metrics, details = evaluate(frame, metric=args.similarity)
    details.update(
        source="Local Steam CSV" if args.csv else "Synthetic demonstration",
        players=frame.shape[0],
        games=frame.shape[1],
        observed=int(frame.notna().sum().sum()),
    )
    args.output.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(args.output / "metrics.csv", index=False)
    (args.output / "evaluation.json").write_text(json.dumps(details, indent=2), encoding="utf-8")
    scores = predict(frame, metric=args.similarity)["Hybrid"]
    recommend(frame, scores, frame.index[0]).to_csv(
        args.output / "recommendations.csv", index=False
    )
    print(json.dumps(details, indent=2))
    print(metrics.to_string(index=False))


if __name__ == "__main__":
    main()
