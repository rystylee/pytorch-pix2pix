import argparse
import time

import torch

from PIL import Image
import cv2

from model import Generator
from util import get_input_tensor, toPIL, denormalize


def test_simple(args):
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

    G = Generator(args.input_nc, args.output_nc, args.ngf).to(device)
    G.load_state_dict(torch.load(args.modelG_state_path, map_location=lambda storage, loc: storage))
    G.eval()

    input_img = Image.open(args.input_img_path).convert('RGB').resize((args.img_size, args.img_size), Image.BICUBIC)
    input_tensor = get_input_tensor(input_img).unsqueeze(0).to(device)

    with torch.no_grad():
        now = time.time()
        out = G(input_tensor)
        end = time.time()
        print('elapsed: {}'.format(end - now))

    out_denormalized = denormalize(out.squeeze()).cpu()
    out_img = toPIL(out_denormalized)
    out_img.show()


def test_recursive(args):
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

    G = Generator(args.input_nc, args.output_nc, args.ngf).to(device)
    G.load_state_dict(torch.load(args.modelG_state_path, map_location=lambda storage, loc: storage))
    G.eval()

    input_img = Image.open(args.input_img_path).convert('RGB').resize((args.img_size, args.img_size), Image.BICUBIC)
    input_tensor = get_input_tensor(input_img).unsqueeze(0).to(device)

    cap = cv2.VideoCapture(0)
    while cap.isOpened():
        now = time.time()
        with torch.no_grad():
            out = G(input_tensor)

            out_denormalized = denormalize(out.squeeze())
            out_denormalized = out_denormalized.cpu().numpy().transpose(1, 2, 0)
            out_denormalized = out_denormalized[:, :, ::-1]
            cv2.imshow('Result', out_denormalized)

            input_tensor = out

        end = time.time()
        elapsed = end - now
        print('elapsed: {}'.format(elapsed))
        print('fps: {}'.format(1.0 / elapsed))

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser()

    # Test option
    parser.add_argument('--modelG_state_path', type=str, default=None)
    parser.add_argument('--input_img_path', type=str, default='imgs/test.png')
    parser.add_argument('--test_mode', type=str, default='simple', choices=['simple', 'recursive'])
    parser.add_argument('--img_size', type=int, default=256)

    # Model
    parser.add_argument('--input_nc', type=int, default=3, help='')
    parser.add_argument('--output_nc', type=int, default=3, help='')
    parser.add_argument('--ngf', type=int, default=64, help='')

    args = parser.parse_args()
    print(args)

    if args.modelG_state_path is not None:
        if args.test_mode == 'simple':
            test_simple(args)
        elif args.test_mode == 'recursive':
            test_recursive(args)
    else:
        print('[!] Could not find the model data...')


if __name__ == "__main__":
    main()
