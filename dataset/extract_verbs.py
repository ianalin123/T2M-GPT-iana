import os
from tqdm import tqdm
import argparse


def main(texts_dir: str, output_path: str | None = None) -> None:
    labels = {}

    # Default output path: same parent directory as texts_dir, file name "verbs.txt"
    if output_path is None:
        parent_dir = os.path.dirname(os.path.abspath(texts_dir))
        output_path = os.path.join(parent_dir, "verbs.txt")

    for filename in tqdm(os.listdir(texts_dir)):
        file_path = os.path.join(texts_dir, filename)
        if not os.path.isfile(file_path):
            continue

        with open(file_path) as f:
            content = f.read().strip().split("\n")

        verbs = set()

        for line in content:
            tokens = line.split("#")[1].split(" ")  # just handling the formatting dictated by each txt file
            for token in tokens:
                if "/VERB" not in token:
                    continue
                verbs.add(token.strip("/VERB"))

        # ordering each verb alphabetically ascendingly since order doesn't matter
        # in distinguishing labels (i.e. walk-run == run-walk)
        label = "-".join(sorted(list(verbs)))

        labels[filename.strip(".txt")] = label

    with open(output_path, "w") as f:
        f.write("\n".join(map(lambda k: f"{k} {labels[k]}", labels)))
    print(f"FileID to verb mapping saved in {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract verbs from HumanML3D text files.")
    parser.add_argument(
        "texts_dir",
        help="Path to the directory containing HumanML3D text files (e.g. dataset/HumanML3D/texts).",
    )
    parser.add_argument(
        "--output",
        "-o",
        default=None,
        help="Output file path for the verbs mapping (default: <parent_of_texts_dir>/verbs.txt).",
    )

    args = parser.parse_args()
    main(args.texts_dir, args.output)