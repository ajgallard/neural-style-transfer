import torch
import argparse

def main():
    parser = argparse.ArgumentParser(description="cuda check parser")
    parser.add_argument("-c", "--cuda",
                        type=int, required=True,
                        help="set it to 1 for running on GPU, 0 for CPU")

    args = parser.parse_args()

    device = torch.device("cuda" if args.cuda else "cpu")
    print(f'Current device setting: {device}', '\n')

    if args.cuda and not torch.cuda.is_available():
        print("ERROR: cuda is not available, try running on CPU")
        sys.exit(1)

if __name__ == '__main__':
    main()