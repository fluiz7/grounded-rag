"""Command-line interface.

    python -m src.cli ingest ./papers          # index PDFs
    python -m src.cli ask "your question"      # ask with citations
    python -m src.cli ask "..." --verbose      # also show critic verdict & retries

Exit codes: 0 verified answer · 3 answer could not be verified · 4 documents don't contain the answer.
"""
import argparse

from .graph import build_graph
from .ingest import ingest


def main() -> None:
    parser = argparse.ArgumentParser(prog="grounded-rag", description="Multi-agent RAG with a groundedness critic (local, via Ollama).")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_ingest = sub.add_parser("ingest", help="Index a PDF file or a folder of PDFs")
    p_ingest.add_argument("path", help="PDF file or folder")

    p_ask = sub.add_parser("ask", help="Ask a question against the indexed documents")
    p_ask.add_argument("question", help="Your question")
    p_ask.add_argument("--verbose", action="store_true", help="Show critic verdict and retry count")

    args = parser.parse_args()

    if args.cmd == "ingest":
        n_pdfs, n_chunks = ingest(args.path)
        print(f"Indexed {n_pdfs} PDF(s) into {n_chunks} chunks.")
        return

    graph = build_graph()
    state = graph.invoke({"question": args.question})
    verified = bool(state.get("grounded"))

    print("\n=== ANSWER " + "=" * 50)
    print(state["answer"])
    if state.get("found") is False:
        tried = state.get("queries_tried", [])
        print(f"\n(i) The documents don't contain the answer — searched {len(tried)} time(s): "
              + " | ".join(tried))
    elif state.get("critic_error"):
        print("\n(!) Not verified: the critic model returned no verdict, so the answer could not be "
              "checked. Try another RAG_CRITIC_MODEL. Use --verbose to inspect the draft.")
    elif not verified:
        print(
            f"\n(!) Not verified: the critic approved no answer in {state.get('attempts', 1)} "
            "retrieval round(s), so none is given. Use --verbose to inspect the unverified draft."
        )

    print("\n=== SOURCES " + "=" * 49)
    for i, d in enumerate(state.get("documents", []), start=1):
        print(f"  [{i}] {d.metadata.get('source', '?')} — page {d.metadata.get('page', '?')}")

    if args.verbose:
        print("\n=== PIPELINE " + "=" * 48)
        print(f"  retrieval rounds : {state.get('attempts', 1)}")
        print(f"  grounded         : {verified}")
        print(f"  found in docs    : {state.get('found') is not False}")
        for claim in state.get("unsupported", []):
            print(f"  unsupported      : {claim}")
        print(f"  critic verdict   :\n{state.get('critique', '(none)')}")
        if state.get("draft"):
            print("\n=== UNVERIFIED DRAFT (do not trust) " + "=" * 25)
            print(state["draft"])

    if state.get("found") is False:
        raise SystemExit(4)
    raise SystemExit(0 if verified else 3)


if __name__ == "__main__":
    main()
