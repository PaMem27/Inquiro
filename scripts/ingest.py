from inquiro.corpus import ingest_papers


def main() -> None:
    chunk_count, paper_count = ingest_papers()
    print(f"Ingested {chunk_count} chunks from {paper_count} papers.")


if __name__ == "__main__":
    main()
