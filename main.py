import argparse
from train import train
from visualize import visualize

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["train", "visualize"], default="train")
    parser.add_argument("--episodes", type=int, default=1000)
    args = parser.parse_args()
    
    if args.mode == "train":
        train(args.episodes)  # Now passes the argument correctly
    elif args.mode == "visualize":
        visualize()
