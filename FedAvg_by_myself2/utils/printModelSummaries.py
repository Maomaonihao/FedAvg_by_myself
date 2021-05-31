import torch
from torch.utils.tensorboard import SummaryWriter
from torchsummary import summary
from models.MNIST.mnist_cnn import MNIST_CNN
from models.MNIST.mnist_mlp import MNIST_MLP

def print_model_summaries(model_name):
    if model_name == "mlp":
        mnist_mlp = MNIST_MLP()
        if torch.cuda.is_available():
            mnist_mlp.cuda()
        print("MNIST MLP SUMMARY:")
        print(summary(mnist_mlp, (28, 28)))

    if model_name == "cnn":
        mnist_cnn = MNIST_CNN()
        if torch.cuda.is_available():
            mnist_cnn.cuda()
        print("\nMNIST CNN SUMMARY:")
        print(summary(mnist_cnn, (1, 28, 28)))