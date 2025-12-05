# file: check_processed_structure.py
# !/usr/bin/env python3
"""Check the structure of data/processed/"""

from pathlib import Path


def check_structure():
    processed_dir = Path('data/processed')

    print("Checking data/processed/ structure:")
    print("=" * 60)

    if not processed_dir.exists():
        print("❌ data/processed/ does not exist!")
        return

    # Check immediate subdirectories
    subdirs = [d for d in processed_dir.iterdir() if d.is_dir()]

    print(f"\nFound {len(subdirs)} subdirectories:")
    for subdir in sorted(subdirs):
        images = list(subdir.glob('*.png')) + list(subdir.glob('*.jpg'))
        print(f"  {subdir.name}: {len(images)} images")

    # Check if there's an extra nesting level
    print("\nFull structure:")
    for item in sorted(processed_dir.rglob('*')):
        if item.is_dir():
            depth = len(item.relative_to(processed_dir).parts)
            indent = "  " * depth
            print(f"{indent}{item.name}/")


if __name__ == "__main__":
    check_structure()