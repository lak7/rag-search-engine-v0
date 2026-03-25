#!/usr/bin/env python3

import argparse
from lib.common import BM25_B, BM25_K1
from lib.keyword_search import bm25_idf_command, bm25_tf_command, build_command, idf_command, search_query, tf_command, tfidf_command


def main() -> None:
    parser = argparse.ArgumentParser(description="Keyword Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    search_parser = subparsers.add_parser("search", help="Search movies using BM25")
    search_parser.add_argument("query", type=str, help="Search query")

    search_parser = subparsers.add_parser("build", help="Build the inverted index")

    search_parser = subparsers.add_parser("tf", help="Term Frequency")
    search_parser.add_argument("doc_id", type=int, help="Search term")
    search_parser.add_argument("term", type=str, help="Search term")

    search_parser = subparsers.add_parser("idf", help="Inverse Term Frequency")
    search_parser.add_argument("term", type=str, help="Search term")

    search_parser = subparsers.add_parser("tfidf", help="TF-IDF")
    search_parser.add_argument("doc_id", type=int, help="Search term")
    search_parser.add_argument("term", type=str, help="Search term")

    bm25_idf_parser = subparsers.add_parser("bm25idf", help="Get BM25 IDF score for a given term")
    bm25_idf_parser.add_argument("term", type=str, help="Term to get BM25 IDF score for")

    bm25_tf_parser = subparsers.add_parser("bm25tf", help="Get BM25 TF score for a given document ID and term")
    bm25_tf_parser.add_argument("doc_id", type=int, help="Document ID")
    bm25_tf_parser.add_argument("term", type=str, help="Term to get BM25 TF score for")
    bm25_tf_parser.add_argument("k1", type=float, nargs='?', default=BM25_K1, help="Tunable BM25 K1 parameter")
    bm25_tf_parser.add_argument("b", type=float, nargs='?', default=BM25_B, help="Tunable BM25 b parameter")


    args = parser.parse_args()

    match args.command:
        case "search":
            results = search_query(args.query)
            # print the search query here
            print(f"These are the results {results}")
            pass
        case "build":
            build_command()
        case "tf":
            tf_command(args.doc_id, args.term)
        case "idf":
            idf_command(args.term)
        case "tfidf":
            tfidf_command(args.doc_id, args.term)
        case "bm25idf":
            bm25_idf_command(args.term)
        case "bm25tf":
            bm25_tf_command(args.doc_id, args.term)
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()