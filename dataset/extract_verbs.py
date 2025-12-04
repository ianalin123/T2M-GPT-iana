import os
import zipfile
from tqdm import tqdm

labels = {}
texts_dir = "dataset/KIT-ML/texts"

for filename in tqdm(os.listdir(texts_dir)):
    with open(os.path.join(texts_dir, filename)) as f:
        content = f.read().strip().split("\n")

    verbs = set()

    for line in content:
        # Some KIT text files contain header / empty lines without a '#' separator.
        # For those lines we simply skip verb extraction.
        if "#" not in line:
            continue

        # Just handling the formatting dictated by each txt file:
        # "<meta stuff> # token1/POS token2/POS ..."
        tokens = line.split("#", 1)[1].split(" ")
        for token in tokens:
            if "/VERB" not in token:
                continue
            verbs.add(token.strip("/VERB"))

    label = "-".join(sorted(list(verbs))) # ordering each verb alphabetically ascendingly since order doesn't matter in distinguishing labels (i.e. walk-run == run-walk)

    labels[filename.strip(".txt")] = label

with open("dataset/KIT-ML/verbs.txt", "w") as f:
    f.write("\n".join(map(lambda k: f"{k} {labels[k]}", labels)))
    print("FileID to verb mapping saved in KIT-ML/verbs.txt")