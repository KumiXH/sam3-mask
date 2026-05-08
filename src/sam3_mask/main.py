from sam3_mask.cli import build_overrides, parse_args
from sam3_mask.config.loader import load_config
from sam3_mask.models.factory import build_backend
from sam3_mask.pipeline.runner import run_pipeline


def main() -> int:
    args = parse_args()
    config = load_config(args.config, build_overrides(args))
    backend = build_backend(config)
    summary = run_pipeline(config=config, backend=backend)
    print(
        "processed={processed} empty={empty} skipped={skipped} failed={failed}".format(
            processed=summary.processed,
            empty=summary.empty,
            skipped=summary.skipped,
            failed=summary.failed,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
