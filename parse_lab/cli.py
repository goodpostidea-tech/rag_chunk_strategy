"""CLI for document parsing experiment."""

import argparse
import json
import sys
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

from parse_lab.pipeline import compare_to_dict, run_parse, run_parse_batch
from parse_lab.registry import list_parsers


def cmd_list(args: argparse.Namespace) -> int:
    rows = list_parsers(include_unavailable=args.all)
    if args.json:
        print(json.dumps(rows, ensure_ascii=False, indent=2))
    else:
        for r in rows:
            flag = "ok" if r["available"] else f"need {r.get('install_hint', '')}"
            print(f"  {r['name']:16} {r['file_types']}  [{flag}]  {r['description']}")
    return 0


def cmd_run(args: argparse.Namespace) -> int:
    parsers = args.parsers.split(",") if args.parsers else None
    result = run_parse(args.file, parsers, doc_profile=args.profile)
    out = compare_to_dict(result)
    if args.json:
        print(json.dumps(out, ensure_ascii=False, indent=2))
    else:
        print(f"File: {result.file_path} | Profile: {result.doc_profile}")
        print(f"{'parser':16} {'score':>6} {'complete':>8} {'table':>6} {'heading':>7} {'ms':>8}")
        for row in result.ranking:
            if row.get("error"):
                print(f"{row['parser']:16}   FAIL  {row['error'][:40]}")
                continue
            print(
                f"{row['parser']:16} {row.get('score', 0):6.3f} "
                f"{row.get('text_completeness', 0):8.3f} "
                f"{row.get('table_structure_score', 0):6.3f} "
                f"{row.get('heading_detection_score', 0):7.3f} "
                f"{row.get('elapsed_ms', 0):8.1f}"
            )
    return 0


def cmd_batch(args: argparse.Namespace) -> int:
    from chunk_lab.datasources.local_dir import list_local_files

    types = args.types.split(",") if args.types else [".pdf", ".docx"]
    files = list_local_files(args.dir, file_types=types)
    parsers = args.parsers.split(",") if args.parsers else None
    results = run_parse_batch([f["path"] for f in files], parsers)
    if args.json:
        print(json.dumps([compare_to_dict(r) for r in results], ensure_ascii=False, indent=2))
    else:
        for r in results:
            top = r.ranking[0]["parser"] if r.ranking else "-"
            print(f"{Path(r.file_path).name:30} profile={r.doc_profile:18} best={top}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="RAG document parsing lab")
    sub = parser.add_subparsers(dest="command", required=True)

    p_list = sub.add_parser("list", help="列出解析器")
    p_list.add_argument("--all", action="store_true")
    p_list.add_argument("--json", action="store_true")
    p_list.set_defaults(func=cmd_list)

    p_run = sub.add_parser("run", help="单文件解析对比")
    p_run.add_argument("--file", "-f", required=True)
    p_run.add_argument("--parsers", "-p")
    p_run.add_argument("--profile")
    p_run.add_argument("--json", action="store_true")
    p_run.set_defaults(func=cmd_run)

    p_batch = sub.add_parser("batch", help="目录批量解析")
    p_batch.add_argument("--dir", "-d", required=True)
    p_batch.add_argument("--parsers", "-p")
    p_batch.add_argument("--types")
    p_batch.add_argument("--json", action="store_true")
    p_batch.set_defaults(func=cmd_batch)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
