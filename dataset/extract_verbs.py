import os
import argparse
from tqdm import tqdm


def extract_verbs(texts_dir: str, output_file: str) -> None:
    """
    Extract unique verb labels from all text files in a directory.

    Assumes each line has a '#' and tokens tagged with '/VERB', e.g. 'walk/VERB'.
    """
    labels = {}

    for filename in tqdm(os.listdir(texts_dir), desc="Processing files"):
        full_path = os.path.join(texts_dir, filename)

        # Skip non-files (e.g. directories)
        if not os.path.isfile(full_path):
            continue

        # NEW: only process .txt files
        if not filename.lower().endswith(".txt"):
            continue

        # NEW: be tolerant of bad encodings
        with open(full_path, encoding="utf-8", errors="ignore") as f:
            content = f.read().strip().split("\n")

        verbs = set()

        for line in content:
            # just handling the formatting dictated by each txt file
            if "#" not in line:
                continue

            tokens = line.split("#", 1)[1].split(" ")
            for token in tokens:
                if "/VERB" not in token:
                    continue
                verbs.add(token.strip("/VERB"))

        # ordering each verb alphabetically asc since order doesn't matter
        # (i.e. walk-run == run-walk)
        label = "-".join(sorted(list(verbs)))

        file_id, _ = os.path.splitext(filename)
        labels[file_id] = label

    with open(output_file, "w") as f:
        f.write("\n".join(f"{k} {labels[k]}" for k in labels))
    print(f"FileID to verb mapping saved in {output_file}")


def main():
    parser = argparse.ArgumentParser(
        description="Extract verb labels from a dataset of POS-tagged text files."
    )
    parser.add_argument(
        "--texts_dir",
        type=str,
        default="dataset/HumanML3D/texts",
        help="Directory containing input text files.",
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default="dataset/HumanML3D/verbs.txt",
        help="Path to output mapping file.",
    )

    args = parser.parse_args()

    extract_verbs(args.texts_dir, args.output_file)


if __name__ == "__main__":
    main()