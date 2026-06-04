import argparse
import torch
from infer import infer_per_machine


def main():
    parser = argparse.ArgumentParser(description="DCASE 2026 Task 2 — Unsupervised Anomalous Sound Detection")
    parser.add_argument("--data_type", type=str, default="dev_data",choices=["dev_data", "eval_data"],
                        help="Data mode (default: dev_data)")
    parser.add_argument("--device", type=str, default=None,
                        help="Device for BEATs inference (default: cuda if available else cpu)")
    parser.add_argument("--result_dir", type=str, default="results",
                        help="Directory for output CSV files (default: results)")
    parser.add_argument("--output_pkl", type=str, default=None,
                        help="Optional path to save raw results as pickle")
    parser.add_argument("--threshold", type=float, default=0.85,
                        help="Gamma PPF percentile for decision threshold (default: 0.85)")
    parser.add_argument("--mono", type=bool, default=False,
                        help="Mono audio (default: False)")
    parser.add_argument("--features", type=str, nargs="+",
                        default=["subband", "beats", "sc"],
                        help="Features to use (default: subband beats sc)")
    args = parser.parse_args()
    data_dir=f"data/dcase2026t2/{args.data_type}/raw"
    output_dir=f"{args.result_dir}/{args.data_type}"

    if args.device is None:
        args.device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"data_dir:   {data_dir}")
    print(f"device:     {args.device}")
    print(f"output_dir: {output_dir}")
    print(f"features:   {args.features}")

    infer_per_machine(
        data_dir=data_dir,
        device=args.device,
        output_dir=output_dir,
        threshold=args.threshold,
        features=args.features,
        mono=args.mono,
    )


if __name__ == "__main__":
    main()
