import glob
import random

import torch.utils.data as data
import torchvision.transforms as transforms

from PIL import Image


class AlignedDataset(data.Dataset):
    def __init__(self, data_dir, direction, scale_size, crop_size):
        super(AlignedDataset, self).__init__()

        self.AB_dir = '{}'.format(data_dir)
        self.AB_paths = glob.glob('{}/*'.format(self.AB_dir))
        self.direction = direction
        self.scale_size = scale_size
        self.crop_size = crop_size

        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])

    def __len__(self):
        return len(self.AB_paths)

    def __getitem__(self, index):
        AB_path = self.AB_paths[index]
        AB = Image.open(AB_path).convert('RGB')
        AB = AB.resize((self.scale_size * 2, self.scale_size), Image.BICUBIC)

        w, h = AB.size

        x_max = self.scale_size - self.crop_size
        y_max = self.scale_size - self.crop_size
        w2 = random.randint(0, x_max)
        h2 = random.randint(0, y_max)

        if self.direction == 'AtoB':
            A = AB.crop((w2, h2, w2 + self.crop_size, h2 + self.crop_size))
            B = AB.crop((self.scale_size + w2, h2, self.scale_size + w2 + self.crop_size, h2 + self.crop_size))
        elif self.direction == 'BtoA':
            B = AB.crop((w2, h2, w2 + self.crop_size, h2 + self.crop_size))
            A = AB.crop((self.scale_size + w2, h2, self.scale_size + w2 + self.crop_size, h2 + self.crop_size))

        A = self.transform(A)
        B = self.transform(B)

        return A, B
