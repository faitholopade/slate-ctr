import argparse
import json
from .demo import run


def main():
    parser = argparse.ArgumentParser(description="Synthetic slate CTR demonstration")
    parser.add_argument("command", choices=["demo"])
    parser.parse_args()
    print(json.dumps(run(), indent=2))


if __name__ == "__main__":
    main()
