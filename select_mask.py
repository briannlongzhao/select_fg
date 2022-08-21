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
    parser.add_argument('--target_class', type=str, help="desired target class, must be in label2id")
    parser.add_argument('--dataset', type=str, default="voc", choices=["coco", "voc"])

    # M step
    parser.add_argument('--M_mode', type=str, default="mean", choices=["mean", "gmm_tied", "gmm_full"])
    parser.add_argument('--M_metric', type=str, default="euclidean")
    parser.add_argument('--M_k', type=float, default=0.5)
    parser.add_argument('--M_n_cluster', type=int, default=3, help="only used if M_mode=gmm")
    parser.add_argument('--M_mahalanobis', action="store_true", default=False, help="only used if M_mode=gmm")

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

    for path in args.save_root.glob(f"*/{args.target_class}*"):
        shutil.rmtree(path)

    return args


if __name__ == '__main__':
    args = parse_arguments()
    alg = FEM(args)
    alg.run()
