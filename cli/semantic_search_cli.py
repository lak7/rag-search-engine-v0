#!/usr/bin/env python3

import argparse

from lib.semantic_search import embed_chunks, embed_query_text, embed_text, fixed_size_chunking, search_query, semantic_chunking, verify_embeddings, verify_model

def main():
    parser = argparse.ArgumentParser(description="Semantic Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    subparsers.add_parser("verify", help="Verifies if embedding model loads properly")

    embed_text_parser = subparsers.add_parser("embed_text", help="Embeds text into a vector")
    embed_text_parser.add_argument("text", type=str, help="Text to embed")

    subparsers.add_parser("verify_embeddings", help="Verifies cached movie embeddings")

    subparsers.add_parser("embed_chunks", help="Load or build chunked movie embeddings")

    embed_query_text_parser = subparsers.add_parser("embedquery", help="Embeds text into a vector")
    embed_query_text_parser.add_argument("query", type=str, help="Query to embed")

    search_query_parser = subparsers.add_parser("search", help="Embeds text into a vector")
    search_query_parser.add_argument("query", type=str, help="Query to embed")

    chunk_query_parser = subparsers.add_parser("chunk", help="To chunk")
    chunk_query_parser.add_argument("text", type=str, help="Text to chunk")
    chunk_query_parser.add_argument("--overlap", type=int, help="Text to chunk")

    semantic_chunk_parser = subparsers.add_parser("semantic_chunk", help="To chunk")
    semantic_chunk_parser.add_argument("text", type=str, help="Text to chunk")
    semantic_chunk_parser.add_argument("--overlap", type=int, help="Text to chunk")
    semantic_chunk_parser.add_argument("--chunk_size", type=int, help="Text to chunk")
 
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
        case "chunk":
            fixed_size_chunking(args.text, args.overlap)
        case "semantic_chunk":
            semantic_chunking(args.text, args.overlap, args.chunk_size)
        case "embed_chunks":
            embed_chunks()
        case _:
            parser.print_help()

if __name__ == "__main__":
    main()