"""CLI for retrieval strategy experiment."""

import argparse
import json
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

from chunk_lab.config import get_settings
from chunk_lab.utils import load_text
from retrieval_lab.pipeline import (
    compare_retrieval_methods,
    evaluate_retrieval_experiment,
    retrieval_response_to_dict,
)
from retrieval_lab.registry import list_retrieval_methods
from retrieval_lab.types import RetrievalQAPair


def cmd_list(args: argparse.Namespace) -> int:
    rows = list_retrieval_methods()
    if args.json:
        print(json.dumps(rows, ensure_ascii=False, indent=2))
    else:
        for r in rows:
            print(f"  {r['name']:16} {r['description']}")
    return 0


def cmd_eval(args: argparse.Namespace) -> int:
    text = load_text(args.input)
    qa_data = json.loads(Path(args.qa).read_text(encoding="utf-8"))
    pairs = [RetrievalQAPair.from_dict(item) for item in qa_data]
    cfg = get_settings()
    resp = evaluate_retrieval_experiment(
        text,
        args.method,
        pairs,
        chunk_strategy=args.chunk_strategy,
        top_k=args.top_k or cfg.eval_top_k,
    )
    print(json.dumps(retrieval_response_to_dict(resp), ensure_ascii=False, indent=2))
    return 0


def cmd_compare(args: argparse.Namespace) -> int:
    text = load_text(args.input)
    qa_data = json.loads(Path(args.qa).read_text(encoding="utf-8"))
    pairs = [RetrievalQAPair.from_dict(item) for item in qa_data]
    methods = args.methods.split(",") if args.methods else None
    cfg = get_settings()
    results = compare_retrieval_methods(
        text,
        pairs,
        methods=methods,
        chunk_strategy=args.chunk_strategy,
        top_k=args.top_k or cfg.eval_top_k,
    )
    out = [retrieval_response_to_dict(r) for r in results]
    if args.json:
        print(json.dumps(out, ensure_ascii=False, indent=2))
    else:
        print(f"{'method':16} {'recall':>8} {'prec':>8} {'mrr':>8}")
        for row in out:
            print(
                f"{row['method']:16} {row['context_recall']:8.3f} "
                f"{row['context_precision']:8.3f} {row['mrr']:8.3f}"
            )
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="RAG retrieval strategy lab")
    sub = parser.add_subparsers(dest="command", required=True)

    p_list = sub.add_parser("list")
    p_list.add_argument("--json", action="store_true")
    p_list.set_defaults(func=cmd_list)

    p_eval = sub.add_parser("eval")
    p_eval.add_argument("--input", "-i", required=True)
    p_eval.add_argument("--qa", default="data/experiments/retrieval_qa.json")
    p_eval.add_argument("--method", "-m", default="hybrid")
    p_eval.add_argument("--chunk-strategy", default="recursive_baseline")
    p_eval.add_argument("--top-k", type=int, default=None)
    p_eval.set_defaults(func=cmd_eval)

    p_cmp = sub.add_parser("compare")
    p_cmp.add_argument("--input", "-i", required=True)
    p_cmp.add_argument("--qa", default="data/experiments/retrieval_qa.json")
    p_cmp.add_argument("--methods")
    p_cmp.add_argument("--chunk-strategy", default="recursive_baseline")
    p_cmp.add_argument("--top-k", type=int, default=None)
    p_cmp.add_argument("--json", action="store_true")
    p_cmp.set_defaults(func=cmd_compare)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
