"""CLI for chunk strategy lab."""

import argparse
import json
import sys
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

from chunk_lab.datasources.corpus import (
    build_corpus_documents,
    fetch_paper,
    fetch_papers,
    list_papers,
    load_corpus_text,
)
from chunk_lab.evaluator import evaluate_retrieval, evaluate_retrieval_on_documents
from chunk_lab.pipeline import chunks_to_response_dict, compare_strategies, run_chunk
from chunk_lab.registry import list_strategies
from chunk_lab.stats import stats_to_dict
from chunk_lab.observability import setup_observability, shutdown_observability
from chunk_lab.types import EvalQAPair
from chunk_lab.utils import load_text


def cmd_list(_args: argparse.Namespace) -> int:
    for s in list_strategies():
        print(f"  {s['name']:22} [{s['category']}] {s['description']}")
    return 0


def cmd_chunk(args: argparse.Namespace) -> int:
    text = load_text(args.input) if args.input else args.text
    if not text:
        print("Error: provide --input or --text", file=sys.stderr)
        return 1

    result = run_chunk(
        text,
        args.strategy,
        title=args.title,
        section=args.section,
    )
    out = chunks_to_response_dict(result, max_preview=args.preview)
    if args.json:
        print(json.dumps(out, ensure_ascii=False, indent=2))
    else:
        print(f"Strategy: {result.strategy} ({result.category})")
        print(f"Chunks: {len(result.chunks)} | Elapsed: {result.elapsed_ms}ms")
        print(json.dumps(stats_to_dict(result.stats), ensure_ascii=False, indent=2))
        if out["preview"]:
            print("\n--- Preview ---")
            for i, p in enumerate(out["preview"]):
                print(f"[{i}] ({p['char_length']} chars) {p['content'][:200]}...")
    return 0


def cmd_compare(args: argparse.Namespace) -> int:
    text = load_text(args.input) if args.input else args.text
    if not text:
        print("Error: provide --input or --text", file=sys.stderr)
        return 1

    strategies = args.strategies.split(",") if args.strategies else None
    results = compare_strategies(text, strategies, title=args.title)

    rows = []
    for r in results:
        rows.append(
            {
                "strategy": r.strategy,
                "category": r.category,
                "chunks": r.stats.chunk_count,
                "avg_tokens": round(r.stats.avg_token_estimate, 1),
                "elapsed_ms": r.elapsed_ms,
                "error": r.category == "error",
                "note": r.description if r.category == "error" else "",
            }
        )

    if args.json:
        print(json.dumps(rows, ensure_ascii=False, indent=2))
    else:
        print(f"{'strategy':22} {'chunks':>8} {'avg_tok':>8} {'ms':>8}  note")
        print("-" * 70)
        for row in rows:
            note = row["note"][:30] if row["error"] else ""
            print(
                f"{row['strategy']:22} {row['chunks']:8} {row['avg_tokens']:8} "
                f"{row['elapsed_ms']:8.1f}  {note}"
            )
    return 0


def cmd_papers_list(args: argparse.Namespace) -> int:
    rows = list_papers(arxiv_only=args.arxiv_only)
    if args.json:
        print(json.dumps(rows, ensure_ascii=False, indent=2))
    else:
        print(f"{'id':24} {'arxiv':12} {'cached':6}  title")
        print("-" * 80)
        for r in rows:
            aid = (r.get("arxiv_id") or "-")[:12]
            cached = "yes" if r["cached"] else "no"
            print(f"{r['id']:24} {aid:12} {cached:6}  {r['title'][:40]}")
    return 0


def cmd_papers_fetch(args: argparse.Namespace) -> int:
    if args.id:
        path = fetch_paper(args.id, full_text=args.full_text, force=args.force)
        print(json.dumps({"id": args.id, "status": "ok", "path": str(path)}, ensure_ascii=False))
        return 0

    ids = args.ids.split(",") if args.ids else None
    results = fetch_papers(
        ids,
        all_arxiv=args.all_arxiv,
        full_text=args.full_text,
        force=args.force,
    )
    print(json.dumps(results, ensure_ascii=False, indent=2))
    ok = sum(1 for r in results if r["status"] == "ok")
    print(f"\nFetched {ok}/{len(results)} papers", file=sys.stderr)
    return 0 if ok == len(results) else 1


def cmd_papers_chunk(args: argparse.Namespace) -> int:
    if args.id:
        from chunk_lab.datasources.corpus import paper_to_document

        doc = paper_to_document(args.id, fetch_if_missing=True)
        text = doc.page_content
        title = doc.metadata.get("title", args.id)
    else:
        ids = args.ids.split(",") if args.ids else None
        text = load_corpus_text(ids, fetch_if_missing=not args.no_fetch)
        title = "references_corpus"

    result = run_chunk(text, args.strategy, title=title)
    out = chunks_to_response_dict(result, max_preview=args.preview)
    if args.json:
        print(json.dumps(out, ensure_ascii=False, indent=2))
    else:
        print(f"Paper corpus | Strategy: {result.strategy} | Chunks: {len(result.chunks)}")
        print(json.dumps(stats_to_dict(result.stats), ensure_ascii=False, indent=2))
    return 0


def cmd_papers_eval(args: argparse.Namespace) -> int:
    qa_data = json.loads(Path(args.qa).read_text(encoding="utf-8"))
    pairs = [EvalQAPair.from_dict(item) for item in qa_data]

    if args.id:
        from chunk_lab.datasources.corpus import paper_to_document

        docs = [paper_to_document(args.id, fetch_if_missing=True)]
    else:
        ids = args.ids.split(",") if args.ids else None
        docs = build_corpus_documents(ids, fetch_if_missing=not args.no_fetch)

    resp = evaluate_retrieval_on_documents(
        docs,
        args.strategy,
        pairs,
        top_k=args.top_k,
        eval_mode=args.judge_mode,
        llm_judge=args.llm_judge,
    )
    print(json.dumps(resp.model_dump(), ensure_ascii=False, indent=2))
    return 0


def cmd_eval(args: argparse.Namespace) -> int:
    text = load_text(args.input)
    qa_data = json.loads(Path(args.qa).read_text(encoding="utf-8"))
    pairs = [EvalQAPair.from_dict(item) for item in qa_data]

    resp = evaluate_retrieval(
        text,
        args.strategy,
        pairs,
        top_k=args.top_k,
        title=args.title,
        eval_mode=args.judge_mode,
        llm_judge=args.llm_judge,
    )
    print(json.dumps(resp.model_dump(), ensure_ascii=False, indent=2))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="RAG dynamic chunk strategy lab")
    sub = parser.add_subparsers(dest="command", required=True)

    p_list = sub.add_parser("list", help="List available strategies")
    p_list.set_defaults(func=cmd_list)

    p_chunk = sub.add_parser("chunk", help="Run one strategy")
    p_chunk.add_argument("--strategy", default="recursive_baseline")
    p_chunk.add_argument("--input", "-i", help="Input text file")
    p_chunk.add_argument("--text", "-t", help="Inline text")
    p_chunk.add_argument("--title", default="")
    p_chunk.add_argument("--section", default="")
    p_chunk.add_argument("--preview", type=int, default=3)
    p_chunk.add_argument("--json", action="store_true")
    p_chunk.set_defaults(func=cmd_chunk)

    p_cmp = sub.add_parser("compare", help="Compare multiple strategies")
    p_cmp.add_argument("--input", "-i", required=True)
    p_cmp.add_argument("--text", "-t")
    p_cmp.add_argument("--strategies", "-s", help="Comma-separated strategy names")
    p_cmp.add_argument("--title", default="")
    p_cmp.add_argument("--json", action="store_true")
    p_cmp.set_defaults(func=cmd_compare)

    p_eval = sub.add_parser("eval", help="Retrieval recall@k eval")
    p_eval.add_argument("--input", "-i", required=True)
    p_eval.add_argument("--qa", required=True, help="JSON file with QA pairs")
    p_eval.add_argument("--strategy", default="recursive_baseline")
    p_eval.add_argument("--top-k", type=int, default=None)
    p_eval.add_argument("--title", default="")
    p_eval.add_argument(
        "--llm-judge",
        action="store_true",
        help="Enable LLM-as-Judge (uses CHUNK_EVAL_JUDGE_* env)",
    )
    p_eval.add_argument(
        "--judge-mode",
        choices=["substring", "judge", "both"],
        default=None,
        help="Override eval mode (default: env or substring)",
    )
    p_eval.set_defaults(func=cmd_eval)

    p_papers = sub.add_parser("papers", help="Reference papers corpus")
    papers_sub = p_papers.add_subparsers(dest="papers_command", required=True)

    pp_list = papers_sub.add_parser("list", help="List papers in catalog")
    pp_list.add_argument("--arxiv-only", action="store_true")
    pp_list.add_argument("--json", action="store_true")
    pp_list.set_defaults(func=cmd_papers_list)

    pp_fetch = papers_sub.add_parser("fetch", help="Download/cache papers")
    pp_fetch.add_argument("--id", help="Single paper id")
    pp_fetch.add_argument("--ids", help="Comma-separated paper ids")
    pp_fetch.add_argument("--all-arxiv", action="store_true", help="Fetch all arXiv papers")
    pp_fetch.add_argument("--full-text", action="store_true", help="Download PDF full text")
    pp_fetch.add_argument("--force", action="store_true")
    pp_fetch.set_defaults(func=cmd_papers_fetch)

    pp_chunk = papers_sub.add_parser("chunk", help="Chunk paper corpus")
    pp_chunk.add_argument("--strategy", default="recursive_baseline")
    pp_chunk.add_argument("--id", help="Single paper")
    pp_chunk.add_argument("--ids", help="Comma-separated papers")
    pp_chunk.add_argument("--no-fetch", action="store_true", help="Require cached papers")
    pp_chunk.add_argument("--preview", type=int, default=3)
    pp_chunk.add_argument("--json", action="store_true")
    pp_chunk.set_defaults(func=cmd_papers_chunk)

    pp_eval = papers_sub.add_parser("eval", help="Eval retrieval on paper corpus")
    pp_eval.add_argument("--qa", default="data/papers_qa.json")
    pp_eval.add_argument("--strategy", default="recursive_baseline")
    pp_eval.add_argument("--id")
    pp_eval.add_argument("--ids")
    pp_eval.add_argument("--top-k", type=int, default=None)
    pp_eval.add_argument("--no-fetch", action="store_true")
    pp_eval.add_argument("--llm-judge", action="store_true")
    pp_eval.add_argument("--judge-mode", choices=["substring", "judge", "both"], default=None)
    pp_eval.set_defaults(func=cmd_papers_eval)

    return parser


def main(argv: list[str] | None = None) -> int:
    setup_observability()
    try:
        parser = build_parser()
        args = parser.parse_args(argv)
        return args.func(args)
    finally:
        shutdown_observability()
