import torchvision.transforms as transforms
import numpy as np


def normalize(tensor):
    transform = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    return transform(tensor)


def denormalize(tensor):
    mean = np.asarray([0.5, 0.5, 0.5])
    std = np.asarray([0.5, 0.5, 0.5])
    transform = transforms.Normalize((-1 * mean / std), (1.0 / std))
    return transform(tensor)


def toPIL(tensor):
    transform = transforms.ToPILImage()
    return transform(tensor)


def get_input_tensor(pil_img):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    return transform(pil_img)
