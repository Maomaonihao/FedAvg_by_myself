from torchvision import transforms, datasets
from torch.utils.data import Dataset, DataLoader

# customize transforms
transforms_mnist = transforms.Compose(
    [
        transforms.ToTensor(),     # image -> Tensor(C, H, W) , grey level 0~255 -> 0~1
        transforms.Normalize((0.1307,),(0.3081,))     # 0~1 -> -> -1~1  , accelerate the convergence of models
    ]
)

# minst_data_train and minst_data_test are the subclass of Dataset
mnist_data_train = datasets.MNIST('./data/mnist', train=True, download=True, transform=transforms_mnist)
mnist_data_test = datasets.MNIST('./data/mnist', train=False, download=True, transform=transforms_mnist)


# class_to_char is the dictionary that maps class names to numeric class
classes = np.array(mnist_data_train.class_to_char.values())
classes_test = np.array(minst_data_test.class_to_char.values())
num_classes = len(classes_test)

print("Classes: {} \t Type:" % (classes, type(classes)))
print("Classes_test: {} \t Type:".format(classes, type(classes_test)))

