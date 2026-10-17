"""Flatten the dataset (category/sign/images) into sign/images, as Keras expects."""
import argparse
import shutil
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--src", required=True,
                        help='Path to the "Data" folder containing the 5 category folders')
    parser.add_argument("--dst", default="data/flat_dataset", help="Output folder")
    args = parser.parse_args()

    src, dst = Path(args.src), Path(args.dst)
    dst.mkdir(parents=True, exist_ok=True)

    for category in sorted(p for p in src.iterdir() if p.is_dir()):
        for sign in sorted(p for p in category.iterdir() if p.is_dir()):
            target = dst / sign.name
            if target.exists():  # same sign name in two categories
                target = dst / f"{sign.name}_{category.name}"
            shutil.copytree(sign, target)

    print(f"Done: {len(list(dst.iterdir()))} sign folders in {dst}")


if __name__ == "__main__":
    main()