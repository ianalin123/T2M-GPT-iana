import os
import zipfile
from tqdm import tqdm
import argparse


def extract_verbs(texts_dir: str):
    labels = {}
    texts_dir = os.path.join(texts_dir, "texts")
    # iterate over all txt files in the given texts directory
    for filename in tqdm(os.listdir(texts_dir)):
        file_path = os.path.join(texts_dir, filename)
        if not os.path.isfile(file_path):
            continue

        with open(file_path) as f:
            content = f.read().strip().split("\n")

        verbs = set()

        for line in content:
            # just handling the formatting dictated by each txt file
            tokens = line.split("#")[1].split(" ")
            for token in tokens:
                if "/VERB" not in token:
                    continue
                verbs.add(token.strip("/VERB"))

        # ordering each verb alphabetically ascendingly since order doesn't matter
        label = "-".join(sorted(list(verbs)))
        labels[filename.strip(".txt")] = label

    # save verbs.txt next to the texts directory (e.g. dataset/HumanML3D/verbs.txt)
    output_dir = os.path.dirname(texts_dir)
    output_path = os.path.join(output_dir, "verbs.txt")
    with open(output_path, "w") as f:
        f.write("\n".join(map(lambda k: f"{k} {labels[k]}", labels)))
    print(f"FileID to verb mapping saved in {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Extract verb labels from a dataset texts directory."
    )
    parser.add_argument(
        "--texts_dir",
        type=str,
        default="dataset/HumanML3D",
        help="Path to the dataset texts directory (e.g. dataset/KIT-ML).",
    )
    args = parser.parse_args()

    extract_verbs(args.texts_dir)