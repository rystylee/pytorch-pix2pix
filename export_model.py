import argparse
import torch

from model import Generator


def export(args):
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

    G = Generator(args.input_nc, args.output_nc, args.ngf).to(device)
    G.load_state_dict(torch.load(args.modelG_state_path, map_location=lambda storage, loc: storage))
    print('Succeed to load state dict!')
    G.eval()

    input_tensor = torch.ones(1, args.input_nc, args.img_size, args.img_size).to(device)

    traced_script_module = torch.jit.trace(G, (input_tensor))
    output = traced_script_module(input_tensor)

    if device == 'cpu':
        name = 'Pix2Pix_cpu.pt'
    else:
        name = 'Pix2Pix_gpu.pt'
    traced_script_module.save(name)
    print('Succeed to save traced script module!')


def main():
    parser = argparse.ArgumentParser()

    # Test option
    parser.add_argument('--modelG_state_path', type=str, default=None)
    parser.add_argument('--img_size', type=int, default=256, help='')

    # Model
    parser.add_argument('--input_nc', type=int, default=3, help='')
    parser.add_argument('--output_nc', type=int, default=3, help='')
    parser.add_argument('--ndf', type=int, default=64, help='')
    parser.add_argument('--ngf', type=int, default=64, help='')

    args = parser.parse_args()
    print(args)

    if args.modelG_state_path is not None:
        export(args)
    else:
        print('[!] Could not find the model data...')


if __name__ == "__main__":
    main()
