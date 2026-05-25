"""CLI for vector store benchmark experiment."""

import argparse
import json

from dotenv import load_dotenv

load_dotenv()

from vstore_lab.benchmark import list_vstore_backends, run_vstore_benchmark, vstore_response_to_dict


def cmd_list(args: argparse.Namespace) -> int:
    rows = list_vstore_backends()
    if args.json:
        print(json.dumps(rows, ensure_ascii=False, indent=2))
    else:
        for r in rows:
            opt = " (optional)" if r.get("optional") else ""
            print(f"  {r['provider']}/{r['index_type']}{opt}  {r['description']}")
    return 0


def cmd_bench(args: argparse.Namespace) -> int:
    sizes = [int(s.strip()) for s in args.sizes.split(",") if s.strip()]
    providers = args.providers.split(",") if args.providers else None
    resp = run_vstore_benchmark(vector_counts=sizes, providers=providers)
    out = vstore_response_to_dict(resp)
    if args.json:
        print(json.dumps(out, ensure_ascii=False, indent=2))
    else:
        print(f"{'provider':10} {'index':8} {'n':>8} {'build_ms':>10} {'p50':>8} {'p99':>8} {'mem':>8}")
        for row in out["rows"]:
            if row.get("error"):
                print(f"{row['provider']:10} ERROR: {row['error'][:40]}")
                continue
            mem = row.get("memory_mb") or 0
            print(
                f"{row['provider']:10} {row['index_type']:8} {row['vector_count']:8} "
                f"{row['build_ms']:10.1f} {row['query_p50_ms']:8.2f} "
                f"{row['query_p99_ms']:8.2f} {mem:8.1f}"
            )
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="RAG vector store benchmark lab")
    sub = parser.add_subparsers(dest="command", required=True)

    p_list = sub.add_parser("list")
    p_list.add_argument("--json", action="store_true")
    p_list.set_defaults(func=cmd_list)

    p_bench = sub.add_parser("bench")
    p_bench.add_argument("--sizes", default="1000,10000")
    p_bench.add_argument("--providers")
    p_bench.add_argument("--json", action="store_true")
    p_bench.set_defaults(func=cmd_bench)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
