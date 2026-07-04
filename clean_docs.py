import os
import re

INPUT_FOLDER = "data/raw"
OUTPUT_FOLDER = "data/cleaned"


def clean_file(path):
    with open(path, "r", encoding="utf-8") as f:
        text = f.read()

    if text.startswith("---"):
        parts = text.split("---", 2)
        if len(parts) == 3:
            text = parts[2]

    sections = re.split(r"(```[\s\S]*?```)", text)

    cleaned = []

    for section in sections:

        if section.startswith("```") and section.endswith("```"):
            cleaned.append(section)
            continue

        section = re.sub(r"</?[A-Za-z][^>]*>", "", section)

        section = re.sub(
            r"\[([^\]]+)\]\([^)]+\)",
            r"\1",
            section
        )

        section = re.sub(
            r"https?://[^\s]+",
            "",
            section
        )

        section = re.sub(
            r"www\.[^\s]+",
            "",
            section
        )

        section = re.sub(
            "["
            "\U0001F600-\U0001F64F"
            "\U0001F300-\U0001F5FF"
            "\U0001F680-\U0001F6FF"
            "\U0001F900-\U0001F9FF"
            "\u2600-\u26FF"
            "\u2700-\u27BF"
            "]+",
            "",
            section,
        )

        section = re.sub(r"\n{3,}", "\n\n", section)

        cleaned.append(section)

    return "".join(cleaned).strip()


def main():

    total = 0
    failed = []

    for root, dirs, files in os.walk(INPUT_FOLDER):

        for file in files:

            if not file.endswith((".md", ".mdx")):
                continue

            input_path = os.path.join(root, file)

            relative = os.path.relpath(
                input_path,
                INPUT_FOLDER
            )

            output_path = os.path.join(
                OUTPUT_FOLDER,
                relative
            )

            try:

                cleaned_text = clean_file(input_path)

                os.makedirs(
                    os.path.dirname(output_path),
                    exist_ok=True
                )

                with open(
                    output_path,
                    "w",
                    encoding="utf-8"
                ) as f:
                    f.write(cleaned_text)

                total += 1
                print("cleaned:", relative)

            except Exception as e:

                failed.append(relative)
                print("failed:", relative)
                print(e)

    print("\nTotal cleaned:", total)
    print("Total failed:", len(failed))


if __name__ == "__main__":
    main()