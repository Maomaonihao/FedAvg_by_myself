import numpy as np
from torchvision import transforms, datasets
from torch.utils.data import Dataset, DataLoader

# customize transforms
transforms_mnist = transforms.Compose(
    [
        transforms.ToTensor(),     # image -> Tensor(C, H, W) , grey level 0~255 -> 0~1
        transforms.Normalize((0.1307,),(0.3081,))     # 0~1 -> -> -1~1  , accelerate the convergence of models
    ]
)

def mnist_data():
    # minst_data_train and minst_data_test are the subclass of Dataset
    mnist_data_train = datasets.MNIST('./data/mnist', train=True, download=True, transform=transforms_mnist)
    mnist_data_test = datasets.MNIST('./data/mnist', train=False, download=True, transform=transforms_mnist)

    print("Classes: {} \t Type: {}" % (classes, type(classes)))
    print("Classes_test: {} \t Type: {}".format(classes, type(classes_test)))
    return mnist_data_train, mnist_data_test
