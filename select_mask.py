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
    parser.add_argument('--debug', action="store_true", default=False, help="if set, only run F-EM on 10% of input")
    parser.add_argument('--bsz', type=int, default=16, help="batch size used in computing feature by feature extractor")

    # M step
    parser.add_argument('--M_mode', type=str, default="mean", choices=["mean", "gmm_tied", "gmm_full"])
    parser.add_argument('--M_metric', type=str, default="euclidean")
    parser.add_argument('--M_k', type=float, default=0.5)
    parser.add_argument('--M_n_cluster', type=int, default=None, help="only used if M_mode=gmm, if not set, try to find best n_cluster")
    parser.add_argument('--M_mahalanobis', action="store_true", default=False, help="only used if M_mode=gmm")

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
