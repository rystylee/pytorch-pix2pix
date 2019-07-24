import argparse
import os
from tqdm import tqdm

import torch.utils

from dataset import AlignedDataset
from trainer import Trainer


def train(args):
    print('Loading dataset...')
    data_dir = '{}/{}'.format(args.data_root, args.dataset_name)
    dataset = AlignedDataset(data_dir, args.direction, args.scale_size, args.crop_size)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    print('The number of training images: {}'.format(len(dataset)))

    trainer = Trainer(args)

    global_step = 1
    for epoch in range(1, args.n_epoch + 1):
        print('Epoch: [{}] has started!'.format(epoch))
        for i, (A, B) in tqdm(enumerate(dataloader), desc='', total=len(dataloader)):
            trainer.optimize(A, B, global_step)

            if global_step % args.save_freq == 0:
                trainer.save_weights(args.save_dir, global_step)

            if global_step % args.video_freq == 0:
                trainer.save_video(args.video_dir, global_step)

            global_step += 1


def main():
    parser = argparse.ArgumentParser()

    # Dataset
    parser.add_argument('--data_root', type=str, default='data', help='')
    parser.add_argument('--dataset_name', type=str, default='cityscapes', help='')
    parser.add_argument('--direction', type=str, default='AtoB', help='')
    parser.add_argument('--scale_size', type=int, default=286, help='')
    parser.add_argument('--crop_size', type=int, default=256, help='')
    parser.add_argument('--batch_size', type=int, default=1, help='')

    # Model
    parser.add_argument('--input_nc', type=int, default=3, help='')
    parser.add_argument('--output_nc', type=int, default=3, help='')
    parser.add_argument('--ndf', type=int, default=64, help='')
    parser.add_argument('--ngf', type=int, default=64, help='')

    # Training
    parser.add_argument('--n_epoch', type=int, default=200, help='')
    parser.add_argument('--lr', type=float, default=0.0002, help='')
    parser.add_argument('--beta1', type=float, default=0.5, help='')
    parser.add_argument('--beta2', type=float, default=0.999, help='')
    parser.add_argument('--lambda_l1', type=float, default=100.0, help='')

    #
    parser.add_argument('--log_dir', type=str, default='logs')
    parser.add_argument('--log_freq', type=int, default=100)
    parser.add_argument('--save_dir', type=str, default='result')
    parser.add_argument('--save_freq', type=int, default=100)
    parser.add_argument('--video_dir', type=str, default='videos')
    parser.add_argument('--video_freq', type=int, default=500)

    args = parser.parse_args()
    print(args)

    if not os.path.exists(args.save_dir):
        os.mkdir(args.save_dir)

    if not os.path.exists(args.log_dir):
        os.mkdir(args.log_dir)

    if not os.path.exists(args.video_dir):
        os.mkdir(args.video_dir)

    train(args)


if __name__ == "__main__":
    main()
