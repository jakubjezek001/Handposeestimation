import argparse


def get_experiement_args():
    parser = argparse.ArgumentParser(description="Arguments for using CPU or GPU.")
    parser.add_argument("--gpu", action="store_true")
    args = parser.parse_args()
    return args.gpu
