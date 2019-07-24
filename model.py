import torch
import torch.nn as nn


# ----------------------------------------------------------------
# Utils
# ----------------------------------------------------------------
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
        if hasattr(m, 'bias') and m.bias is not None:
            nn.init.constant_(m.bias.data, 0.0)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0.0)


# ----------------------------------------------------------------
# Blocks
# ----------------------------------------------------------------
class DownBlock(nn.Module):
    def __init__(self, in_dim, out_dim, k=4, s=2, p=1,
                 use_activation=True, use_norm=True, use_bias=False):
        super(DownBlock, self).__init__()

        layers = []
        if use_activation:
            layers.append(nn.LeakyReLU(0.2, inplace=True))
        layers.append(nn.Conv2d(in_dim, out_dim, kernel_size=k, stride=s, padding=p, bias=use_bias))
        if use_norm:
            layers.append(nn.BatchNorm2d(out_dim))

        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


class UpBlock(nn.Module):
    def __init__(self, in_dim, out_dim, k=4, s=2, p=1,
                 use_activation=True, use_norm=True, use_dropout=False, use_bias=False):
        super(UpBlock, self).__init__()

        layers = []
        if use_activation:
            layers.append(nn.ReLU(inplace=True))
        layers.append(nn.ConvTranspose2d(in_dim, out_dim, kernel_size=k, stride=s, padding=p, bias=use_bias))
        if use_norm:
            layers.append(nn.BatchNorm2d(out_dim))
        if use_dropout:
            layers.append(nn.Dropout(0.5))

        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


# ----------------------------------------------------------------
# Generator
# ----------------------------------------------------------------
class Generator(nn.Module):
    def __init__(self, input_nc, output_nc, ngf):
        super(Generator, self).__init__()

        self.down1 = DownBlock(input_nc, ngf, use_activation=False, use_norm=False)
        self.down2 = DownBlock(ngf, ngf * 2)
        self.down3 = DownBlock(ngf * 2, ngf * 4)
        self.down4 = DownBlock(ngf * 4, ngf * 8)
        self.down5 = DownBlock(ngf * 8, ngf * 8)
        self.down6 = DownBlock(ngf * 8, ngf * 8)
        self.down7 = DownBlock(ngf * 8, ngf * 8)
        self.down8 = DownBlock(ngf * 8, ngf * 8, use_norm=False)

        self.up1 = UpBlock(ngf * 8, ngf * 8, use_dropout=True)
        self.up2 = UpBlock(ngf * 16, ngf * 8, use_dropout=True)
        self.up3 = UpBlock(ngf * 16, ngf * 8, use_dropout=True)
        self.up4 = UpBlock(ngf * 16, ngf * 8)
        self.up5 = UpBlock(ngf * 16, ngf * 4)
        self.up6 = UpBlock(ngf * 8, ngf * 2)
        self.up7 = UpBlock(ngf * 4, ngf)
        self.up8 = UpBlock(ngf * 2, output_nc, use_norm=False, use_bias=True)

        self.tanh = nn.Tanh()

    def forward(self, x):
        down1 = self.down1(x)
        down2 = self.down2(down1)
        down3 = self.down3(down2)
        down4 = self.down4(down3)
        down5 = self.down5(down4)
        down6 = self.down6(down5)
        down7 = self.down7(down6)
        down8 = self.down8(down7)

        up1 = self.up1(down8)
        up2 = self.up2(torch.cat([up1, down7], dim=1))
        up3 = self.up3(torch.cat([up2, down6], dim=1))
        up4 = self.up4(torch.cat([up3, down5], dim=1))
        up5 = self.up5(torch.cat([up4, down4], dim=1))
        up6 = self.up6(torch.cat([up5, down3], dim=1))
        up7 = self.up7(torch.cat([up6, down2], dim=1))
        up8 = self.up8(torch.cat([up7, down1], dim=1))

        out = self.tanh(up8)
        return out


# ----------------------------------------------------------------
# Discriminator
# ----------------------------------------------------------------
class Discriminator(nn.Module):
    def __init__(self, input_nc, ndf):
        super(Discriminator, self).__init__()

        self.model = nn.Sequential(
            self._layer(input_nc, ndf, use_norm=False),
            self._layer(ndf, ndf * 2),
            self._layer(ndf * 2, ndf * 4),
            self._layer(ndf * 4, ndf * 8, s=1),
            self._layer(ndf * 8, 1, s=1, use_norm=False, use_activation=False),
        )

    def _layer(self, in_dim, out_dim, k=4, s=2, p=1, use_norm=True, use_activation=True):
        layers = []
        layers.append(nn.Conv2d(in_dim, out_dim, kernel_size=k, stride=s, padding=p))
        if use_norm:
            layers.append(nn.BatchNorm2d(out_dim))
        if use_activation:
            layers.append(nn.LeakyReLU(0.2, inplace=True))

        return nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)
