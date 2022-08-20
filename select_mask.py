import argparse
import shutil
import warnings
from pathlib import Path

from FEM import FEM

warnings.filterwarnings("ignore")


def parse_arguments():
    """Parses the arguments passed to the run.py script."""
    parser = argparse.ArgumentParser()

    parser.add_argument('--img_root', type=str,
                        default="./img_root")
    parser.add_argument('--mask_root', type=str,
                        default="./VOCmask_entseg")
    parser.add_argument('--save_root', type=str,
                        default="./save_root")

    parser.add_argument('--num_iter', type=int, default=2)
    parser.add_argument('--target_class', type=str)
    parser.add_argument('--dataset', type=str, default="voc", choices=["coco", "voc"])

    # TODO
    parser.add_argument('--pick', type=int, default=50)  # select 200
    parser.add_argument('--pick_step', type=int, default=100)
    parser.add_argument('--ignore_small', type=float, default=0.01)
    parser.add_argument('--ignore_large', type=float, default=0.8)
    parser.add_argument('--classifier_thresh', type=float, default=0.5)
    parser.add_argument('--method', type=str, default="gmm", choices=["em", "gmm"])

    args = parser.parse_args()

    # additional op
    args.save_root = Path(args.save_root)
    args.mask_root = Path(args.mask_root)
    args.img_root = Path(args.img_root)

    if (args.save_root / args.target_class).exists():
        shutil.rmtree(args.save_root / args.target_class)
    if (args.save_root / f"{args.target_class}_mask").exists():
        shutil.rmtree(args.save_root / f"{args.target_class}_mask")

    return args


if __name__ == '__main__':
    args = parse_arguments()
    alg = FEM(args)
    alg.run()
