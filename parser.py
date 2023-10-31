import os
import argparse


def make_parser():
    parser = argparse.ArgumentParser(description="DiceGNN arguments")
    parser.add_argument("--dataset_name",
                        help="Dataset Name",
                        type=str,
                        choices=('ogbn-papers100M', 'uk', 'twitter'),
                        default='twitter')

    parser.add_argument("--dataset_root",
                        help="Dataset root path",
                        type=str, default=f'{os.environ["root_dir"]}/{os.environ["USER"]}/datasets')

    parser.add_argument("--profs",
                        help="existing profiling infos (t_cpu (in ms), t_dma, t_gpu, t_model, total_batches), leave blank to re-profile",
                        type=str, default='23.95,10.12,25.4,4.54,1179')

    parser.add_argument("--num_workers",
                        help="Number of cpu batching workers",
                        type=int, default=8)

    parser.add_argument("--buffer_size",
                        help="gpu buffer size",
                        type=int, default=20)

    parser.add_argument("--hidden_features",
                        help="Number of hidden features",
                        type=int, default=256)

    parser.add_argument("--lr",
                        help="Learning rate",
                        type=float, default=0.003)

    parser.add_argument("--trials",
                        help="Number of trials to run profiling",
                        type=int, default=1)

    parser.add_argument("--epochs",
                        help="Number of epochs to run training",
                        type=int, default=5)

    parser.add_argument("--train_batch_size",
                        help="Size of training batches",
                        type=int, default=1024)

    parser.add_argument("--train_fanouts",
                        help="Training fanouts",
                        type=int, default=[5, 10, 15], nargs="*")

    parser.add_argument("--CPU_batcher",
                        help="CPU batching method",
                        type=str,
                        choices=("salient"),
                        default="salient")

    parser.add_argument("--GPU_batcher",
                        help="GPU batching method",
                        type=str,
                        choices=("dgl"),
                        default="dgl")
    return parser
