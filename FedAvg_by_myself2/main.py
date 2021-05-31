import numpy as np
import torch
import torch.nn as nn
from utils.partitionData import iid_partition, non_iid_partition
from utils.loadDatasets import mnist_data
from training.server import server_training
from testing import testing
from models.MNIST.mnist_cnn import MNIST_CNN
from models.MNIST.mnist_mlp import MNIST_MLP
from utils.printModelSummaries import print_model_summaries
import os

os.chdir('D:\\gitRepos\\FedAvg\\FedAvg_by_myself2\\results')

# load  dataset MNIST
mnist_data_train, mnist_data_test = mnist_data()

classes = np.array(list(mnist_data_train.class_to_idx.values()))   #class_to_char is the dictionary that maps class names to numeric class
classes_test = np.array(list(mnist_data_test.class_to_idx.values()))
num_classes = len(classes_test)
print("Classes: {} \t Type: {}" % (classes, type(classes)))
print("Classes_test: {} \t Type: {}".format(classes, type(classes_test)))

# loss function
criterion = nn.CrossEntropyLoss()

print_model_summaries("cnn")
# ============================ MNIST CNN on IID =================================

# Training

rounds = 100       # number of training rounds
C = 0.1         # client fraction
K = 100       # number of clients
E = 5        # number of training passes on local dataset for each round
batch_size = 10       # batch size
lr=0.01         # learning Rate

# data partition dictionary
iid_dict = iid_partition(mnist_data_train, 100)

mnist_cnn = MNIST_CNN()   # load model

if torch.cuda.is_available():
  mnist_cnn.cuda()

print("MNIST CNN on IID training begins:")
mnist_cnn_iid_trained = server_training(mnist_cnn, rounds, batch_size, lr, mnist_data_train, iid_dict, C, K, E, "MNIST CNN on IID Dataset", "orange")

# Testing
print("MNIST CNN on IID testing begins:")
testing(mnist_cnn_iid_trained, mnist_data_test, 128, criterion, num_classes, classes_test)

# ============================ ## MNIST CNN on Non IID ==============================

# Training
data_dict = non_iid_partition(mnist_data_train, 100, 200, 300, 2)
mnist_cnn = MNIST_CNN()

if torch.cuda.is_available():
  mnist_cnn.cuda()

print("MNIST CNN on Non IID training begins:")
mnist_cnn_non_iid_trained = server_training(mnist_cnn, rounds, batch_size, lr, mnist_data_train, data_dict, C, K, E, "MNIST CNN on Non-IID Dataset", "green")

# Testing
print("MNIST CNN on Non IID testing begins:")
testing(mnist_cnn_non_iid_trained, mnist_data_test, 128, criterion, num_classes, classes_test)


print_model_summaries("mlp")
# ============================ ## MNIST MLP on IID ============================

# Training
rounds = 100
C = 0.1
K = 100
E = 5
batch_size = 10
lr=0.03
data_dict = iid_partition(mnist_data_train, 100)
mnist_mlp = MNIST_MLP()

if torch.cuda.is_available():
  mnist_mlp.cuda()

print("MNIST MLP on IID training begins:")
mnist_mlp_iid_trained = server_training(mnist_mlp, rounds, batch_size, lr, mnist_data_train, data_dict, C, K, E, "MNIST MLP on IID Dataset", "orange")

# Testing
print("MNIST MLP on IID testing begins:")
testing(mnist_mlp_iid_trained, mnist_data_test, 128, criterion, num_classes, classes_test)

# ============================ ## MNIST MLP on Non IID ============================

# Training
lr=0.05
data_dict = non_iid_partition(mnist_data_train, 100, 200, 300, 2)
mnist_mlp = MNIST_MLP()

if torch.cuda.is_available():
  mnist_mlp.cuda()

print("MNIST MLP on Non IID training begins:")
mnist_mlp_non_iid_trained = server_training(mnist_mlp, rounds, batch_size, lr, mnist_data_train, data_dict, C, K, E, "MNIST MLP on Non-IID Dataset", "green")

# Testing
print("MNIST MLP on Non IID testing begins:")
testing(mnist_mlp_non_iid_trained, mnist_data_test, 128, criterion, num_classes, classes_test)