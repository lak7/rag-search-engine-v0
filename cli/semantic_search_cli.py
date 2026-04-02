#!/usr/bin/env python3

import argparse

from lib.semantic_search import embed_query_text, embed_text, search_query, verify_embeddings, verify_model

def main():
    parser = argparse.ArgumentParser(description="Semantic Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    subparsers.add_parser("verify", help="Verifies if embedding model loads properly")

    embed_text_parser = subparsers.add_parser("embed_text", help="Embeds text into a vector")
    embed_text_parser.add_argument("text", type=str, help="Text to embed")

    subparsers.add_parser("verify_embeddings", help="Verifies cached movie embeddings")

    embed_query_text_parser = subparsers.add_parser("embedquery", help="Embeds text into a vector")
    embed_query_text_parser.add_argument("query", type=str, help="Query to embed")

    search_query_parser = subparsers.add_parser("search", help="Embeds text into a vector")
    search_query_parser.add_argument("query", type=str, help="Query to embed")
 
    args = parser.parse_args()

    match args.command:
        case "verify":
            verify_model()
        case "embed_text":
            embed_text(args.text)
        case "verify_embeddings":
            verify_embeddings()
        case "embedquery":
            embed_query_text(args.query)
        case "search":
            search_query(args.query)
        case _:
            parser.print_help()

if __name__ == "__main__":
    main()