import argparse

from inquiro.graph import answer_question
from inquiro.utils import source_names


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="inquiro",
        description="Ask Inquiro a question about your research corpus.",
    )
    parser.add_argument(
        "question",
        nargs="?",
        default="What are the main methods discussed in these papers?",
    )
    args = parser.parse_args()

    result = answer_question(args.question)

    print(f"Route: {result['route']}")
    print("\nAnswer:\n")
    print(result["answer"])
    print("\nEvaluation:")
    print(f"{result['eval_score']} - {result['eval_feedback']}")
    print("\nSources:")
    for source in source_names(result["docs"]):
        print(f"- {source}")


if __name__ == "__main__":
    main()
