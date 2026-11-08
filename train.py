import argparse

import yaml

from weakflow.trainer import train


def main():
    parser = argparse.ArgumentParser()

    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--config")
    source.add_argument("--resume")

    parser.add_argument("--data-root")
    parser.add_argument("--output-root")
    parser.add_argument("--total-steps", type=int)
    parser.add_argument("--device", choices=["cuda", "cpu"])
    parser.add_argument("--smoke", action="store_true")

    args = parser.parse_args()
    cfg = None

    if args.config:
        with open(args.config, encoding="utf-8") as f:
            cfg = yaml.safe_load(f)

        if args.output_root:
            cfg["output_root"] = args.output_root

    elif args.output_root:
        parser.error(
            "再開時の保存先はチェックポイントと同じフォルダです"
        )

    train(
        cfg=cfg,
        resume=args.resume,
        total_steps=args.total_steps,
        data_root=args.data_root,
        device=args.device,
        smoke=args.smoke,
    )


if __name__ == "__main__":
    main()