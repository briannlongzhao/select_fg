import argparse
import shutil
import warnings
from pathlib import Path

from FEM import FEM

warnings.filterwarnings("ignore")


def parse_arguments():
    """Parses the arguments passed to the run.py script."""
    parser = argparse.ArgumentParser()

    parser.add_argument('--img_root', type=str, required=True, help="root directory that contains original images of each class")
    parser.add_argument('--mask_root', type=str, required=True, help="root directory that contains .npy ent seg mask of each original image of each class")
    parser.add_argument('--save_root', type=str, default="./saved", help="where the output is saved")

    parser.add_argument('--num_iter', type=int, default=2, help="number of iterations should F-EM run")
    parser.add_argument('--target_class', type=str, help="desired target class, must be in label2id (see metadata directory)")
    parser.add_argument('--dataset', type=str, default="voc", choices=["coco", "voc"], help="which dataest F-EM is run on")
    parser.add_argument('--bsz', type=int, default=16, help="batch size used in computing feature by feature extractor")
    parser.add_argument('--load_step1', action="store_true", default=False, help="if set, don't compute step 1 but load from @save_root")
    parser.add_argument('--load_embs', action="store_true", default=False, help="if set, don't compute embeddings but load from @save_root")
    parser.add_argument('--filter_thresh', type=float, default=-1.0, help="classification score threshold to filter final result")
    parser.add_argument('--ignore_person', action="store_true", default=False, help="ignore person segments")

    parser.add_argument('--debug', action="store_true", default=False, help="if set, only run F-EM on 10 percent of input")
    # M step
    parser.add_argument('--M_mode', type=str, default="mean", choices=["mean", "gmm_tied", "gmm_full"], help="in M step, which algorithm should be run to determine top k percent")
    parser.add_argument('--M_metric', type=str, default="euclidean", help="in M step, which distance metric is used in selecting top k percent")
    parser.add_argument('--M_k', type=float, default=0.5, nargs='+', help="in M step, retain k percent of total images")
    parser.add_argument('--M_n_cluster', type=int, default=None, help="only used if M_mode=gmm, if not set, try to find best n_cluster")

    args = parser.parse_args()

    # additional op
    args.save_root = Path(args.save_root)
    args.save_root = args.save_root / args.dataset / f"{args.M_mode}[{args.M_metric},{args.M_k}]"
    args.mask_root = Path(args.mask_root)
    args.img_root = Path(args.img_root)

    # for path in args.save_root.glob(f"*/{args.target_class}*"):
    #     shutil.rmtree(path)

    return args


if __name__ == '__main__':
    args = parse_arguments()
    alg = FEM(args)
    alg.run()
