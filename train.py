import os
from solver import Solver
import numpy as np
import random
import argparse
import torch.multiprocessing

torch.multiprocessing.set_sharing_strategy('file_system')


def set_seed(seed=2023):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    #torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def parse_args():

    parser = argparse.ArgumentParser()

    parser.add_argument('--ckpt_root', type=str, default='./ckpt/ckpt_res2023/',
                        help='The root for saving your checkpoint. ')

    parser.add_argument('--train_init_epoch', type=int, default=0, help='init_epoch.')

    parser.add_argument('--train_end_epoch', type=int, default=70, help='end_epoch.')

    parser.add_argument('--train_doc_path', type=str, default='./training_record/training_res2023.txt',
                        help='The root for saving your training log.')

    parser.add_argument('--learning_rate', type=float, default=1e-5, help='learning_rate.')

    parser.add_argument('--weight_decay', type=float, default=1e-4, help='weight_decay.')

    parser.add_argument('--milestones', type=list, default=[50, 60], help='milestones for learning rate.')

    parser.add_argument('--train_batch_size', type=int, default=10, help='train_batch_size.')

    parser.add_argument('--train_num_thread', type=int, default=1, help='train_num_thread.')

    parser.add_argument('--backbone', type=str, default='vgg16',
                        help='The optional backbones are vgg16 and resnet50.')

    parser.add_argument('--cosal_set', type=str, default='COCO9k',
                        help='The dataset for training co-saliency branch.')

    parser.add_argument('--sal_set', type=str, default='DUTS',
                        help='The dataset for training saliency head.')

    parser.add_argument('--fix_seed', type=bool, default=True,
                        help='Fix your training seed.')

    return parser.parse_args()


def main():
    args = parse_args()
    solver = Solver(backbone=args.backbone)
    # An example to build "train_roots".
    if args.cosal_set == 'COCO9k':
        train_roots = {'img': './Dataset/COCO9213/img/',
                       'gt': './Dataset/COCO9213/gt/',
                       }
    elif args.cosal_set == 'DUTS':
        train_roots = {'img': './Dataset/Jigsaw_DUTS/img/',
                       'gt': './Dataset/Jigsaw_DUTS/gt/',
                       }

    if args.sal_set == 'DUTS':
        sal_root = './Dataset/DUTS-TR'
    elif args.sal_set == 'COCO9k':
        sal_root = './Dataset/COCOSAL'  # COCOSAL and COCO9213 are the same data set, but the arrangement is different.

    if args.fix_seed:
        set_seed()

    solver.train(roots=train_roots,
                 init_epoch=args.train_init_epoch,
                 end_epoch=args.train_end_epoch,
                 learning_rate=args.learning_rate,
                 batch_size=args.train_batch_size,
                 weight_decay=args.weight_decay,
                 ckpt_root=args.ckpt_root,
                 doc_path=args.train_doc_path,
                 num_thread=args.train_num_thread,
                 milestones=args.milestones,
                 sal_root=sal_root,
                 fix_seed=args.fix_seed,
                 pin=False)


if __name__ == '__main__':
    main()
