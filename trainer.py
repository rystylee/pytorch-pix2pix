import os
import time

import torch
import torch.nn.functional as F
import torch.optim as optim
import torchvision.utils
from torch.utils.tensorboard import SummaryWriter

from model import weights_init, Discriminator, Generator
from util import denormalize


def gan_loss(x, target):
    if target == 1:
        label = torch.ones(x.size()).to(x.device)
    elif target == 0:
        label = torch.zeros(x.size()).to(x.device)
    else:
        raise NotImplementedError('[!] The target {] is not found.'.format(target))

    return F.binary_cross_entropy_with_logits(x, label)


def l1_loss(x, target):
    return F.l1_loss(x, target)


class Trainer(object):
    def __init__(self, args):
        self.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

        self.G = Generator(args.input_nc, args.output_nc, args.ngf).to(self.device)
        self.G.apply(weights_init)
        print(self.G)

        self.D = Discriminator(args.input_nc + args.output_nc, args.ndf).to(self.device)
        self.D.apply(weights_init)
        print(self.D)

        self.optim_G = optim.Adam(self.G.parameters(), lr=args.lr, betas=(args.beta1, args.beta2))
        self.optim_D = optim.Adam(self.D.parameters(), lr=args.lr, betas=(args.beta1, args.beta2))

        # Parameters
        self.lambda_l1 = args.lambda_l1
        self.log_freq = args.log_freq

        time_str = time.strftime("%Y%m%d-%H%M%S")
        self.writer = SummaryWriter('{}/{}-{}'.format(args.log_dir, args.dataset_name, time_str))

    def __del__(self):
        self.writer.close()

    def save_weights(self, save_dir, global_step):
        d_name = 'D_{}.pth'.format(global_step)
        g_name = 'G_{}.pth'.format(global_step)

        torch.save(self.D.state_dict(), os.path.join(save_dir, d_name))
        torch.save(self.G.state_dict(), os.path.join(save_dir, g_name))

    def optimize(self, A, B, global_step):
        A = A.to(self.device)
        B = B.to(self.device)

        # Logging the input images
        if global_step % self.log_freq == 0:
            log_real_A = torchvision.utils.make_grid(A)
            log_real_A = denormalize(log_real_A)
            self.writer.add_image('real_A', log_real_A, global_step)

            log_real_B = torchvision.utils.make_grid(B)
            log_real_B = denormalize(log_real_B)
            self.writer.add_image('real_B', log_real_B, global_step)

        # Forward pass
        fake_B = self.G(A)

        if global_step % self.log_freq == 0:
            log_fake_B = torchvision.utils.make_grid(fake_B)
            log_fake_B = denormalize(log_fake_B)
            self.writer.add_image('fake_B', log_fake_B, global_step)

        # ----------------------------------------------------------------
        # 1. Train D
        # ----------------------------------------------------------------
        real_pair = torch.cat([A, B], dim=1)
        real_D = self.D(real_pair)
        loss_real_D = gan_loss(real_D, target=1)

        fake_pair = torch.cat([A, fake_B], dim=1)
        fake_D = self.D(fake_pair.detach())
        loss_fake_D = gan_loss(fake_D, target=0)

        loss_D = (loss_real_D + loss_fake_D) * 0.5
        self.writer.add_scalar('loss/loss_D', loss_D.item(), global_step)

        self.optim_D.zero_grad()
        loss_D.backward()
        self.optim_D.step()

        # ----------------------------------------------------------------
        # 2. Train G
        # ----------------------------------------------------------------
        with torch.no_grad():
            fake_D2 = self.D(fake_pair)

        loss_G_gan = gan_loss(fake_D2, target=1)
        loss_G_l1 = l1_loss(fake_B, B) * self.lambda_l1

        loss_G = loss_G_gan + loss_G_l1
        self.writer.add_scalar('loss/loss_G', loss_G.item(), global_step)

        self.optim_G.zero_grad()
        loss_G.backward()
        self.optim_G.step()