import argparse

from lib.hybrid_search import normalize_list, weighted_search

def main() -> None:
    parser = argparse.ArgumentParser(description="Hybrid Search CLI")
    subparser = parser.add_subparsers(dest="command", help="Available commands")

    normalize_parser = subparser.add_parser("normalize", help="Normalize command to normalize scores")
    normalize_parser.add_argument("scores", type=float, nargs="+", help="List of scores to Normalize")

    weighted_search_parser = subparser.add_parser("weighted_search", help="Search command for weighted search")
    weighted_search_parser.add_argument("query", type=str, help="Query to search")
    weighted_search_parser.add_argument("--alpha", type=float, default=0.5, help="To adjust between bm25 and semantic")
    weighted_search_parser.add_argument("--limit", type=int, default=5, help="Limit to retrieve")

    args = parser.parse_args()

    match args.command:
        case "normalize":
            normalized_scores = normalize_list(args.scores)
            for score in normalized_scores:
                print(f"* {score:.4f}")
        case "weighted_search":
            weighted_search(args.query, args.alpha, args.limit)
        case _:
            parser.print_help()

if __name__ == "__main__":
    main()