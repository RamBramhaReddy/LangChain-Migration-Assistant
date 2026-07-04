from pipeline import run_pipeline


def main():

    print()

    print("=" * 100)
    print("LangChain Migration Assistant")
    print("=" * 100)

    while True:

        print()

        query = input(
            "Ask a question or paste your LangChain code:\n\n"
        ).strip()

        if not query:

            print("\nGoodbye!")
            break

        try:

            answer = run_pipeline(query)

            print()
            print("=" * 100)
            print("ANSWER")
            print("=" * 100)
            print()

            print(answer)

        except KeyboardInterrupt:

            print("\n\nInterrupted.")
            break

        except Exception as e:

            print()
            print("=" * 100)
            print("ERROR")
            print("=" * 100)
            print()

            print(e)


if __name__ == "__main__":

    main()