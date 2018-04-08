# following: https://medium.com/ai-society/gans-from-scratch-1-a-deep-introduction-with-code-in-pytorch-and-tensorflow-cb03cdcdba0f
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from utils import Logger

def mnist_data():
    compose = transforms.Compose([transforms.ToTensor(),transforms.Normalize((.5,.5,.5),(.5,.5,.5))])
    out_dir = './dataset'
    return datasets.MNIST(root=out_dir,train=True,transform=compose,download=True)


# Load MNIST data
data = mnist_data()

# Create loader with data, so that we can iterate over it
data_loader = torch.utils.data.DataLoader(data,batch_size=100,shuffle=True)

# Num batches
num_batches = len(data_loader) # this should be 100 !!!


# Disciminator Network class
class DiscriminatorNet(nn.Module):
    """
    A three hidden-layer disciminative neural network
    """
    def __init__(self):
        super(DiscriminatorNet,self).__init__()
        n_features = 784
        n_out = 1

        self.hidden0 = nn.Sequential(nn.Linear(n_features,1024),nn.LeakyReLU(0.2),nn.Dropout(0.3))
        self.hidden1 = nn.Sequential(nn.Linear(1024,512),nn.LeakyReLU(0.2),nn.Dropout(0.3))
        self.hidden2 = nn.Sequential(nn.Linear(512,256),nn.LeakyReLU(0.2),nn.Dropout(0.3))
        self.out = nn.Sequential(nn.Linear(256,n_out),nn.Sigmoid())

    def forward(self,x):
        x = self.hidden0(x)
        x = self.hidden1(x)
        x = self.hidden2(x)
        x = self.out(x)
        return x

disciminator = DiscriminatorNet()

def images_to_vectors(images):
    return images.view(images.size(0),784)

def vectors_to_images(vectors):
    return vectors.view(vectors.size(0),1,28,28)

class GeneratorNet(torch.nn.Module):
    """
    A three hidden-layer generative neural network
    """
    def __init__(self):
        super(GeneratorNet, self).__init__()
        n_features = 100
        n_out = 784

        self.hidden0 = nn.Sequential(
            nn.Linear(n_features, 256),
            nn.LeakyReLU(0.2)
        )
        self.hidden1 = nn.Sequential(
            nn.Linear(256, 512),
            nn.LeakyReLU(0.2)
        )
        self.hidden2 = nn.Sequential(
            nn.Linear(512, 1024),
            nn.LeakyReLU(0.2)
        )

        self.out = nn.Sequential(
            nn.Linear(1024, n_out),
            nn.Tanh()
        )

    def forward(self, x):
        x = self.hidden0(x)
        x = self.hidden1(x)
        x = self.hidden2(x)
        x = self.out(x)
        return x

generator = GeneratorNet()

def noise(size):
    n = Variable(torch.randn(size,100))
    return n

d_optimizer = optim.Adam(disciminator.parameters(),lr=0.0002)
g_optimizer = optim.Adam(generator.parameters(),lr=0.0002)

loss = nn.BCELoss()

def ones_target(size):
    data = Variable(torch.ones(size,1))
    return data

def zeros_target(size):
    data = Variable(torch.zeros(size,1))
    return data

def train_disciminator(optimizer, real_data, fake_data):
    N = real_data.size(0)
    # Reset gradients
    optimizer.zero_grad()

    # 1.1 Train on real data
    prediction_real = disciminator(real_data)
    # calculate error and backpropagate
    error_real = loss(prediction_real,ones_target(N))
    error_real.backward()

    #1.2 Train on Fake data
    prediction_fake = disciminator(fake_data)
    # calculate error and backpropagate
    error_fake = loss(prediction_fake,zeros_target(N))
    error_fake.backward()

    # 1.3 update wieghts with gradients
    optimizer.step()

    # Return error and predictions for real and fake inputs
    return error_real + error_fake,prediction_real,prediction_fake


def train_generator(optimizer,fake_data):
    N = fake_data.size(0)

    # Reset gradients
    optimizer.zero_grad()

    # Samplenoise and generate fake data
    prediction = disciminator(fake_data)

    # Calcualte error and backpropagate
    error = loss(prediction, ones_target(N))
    error.backward()

    # update weights with gradients
    optimizer.step()

    return error

num_test_samples = 16
test_noise = noise(num_test_samples)

# Create logger instance
logger = Logger(model_name='VGAN', data_name='MNIST')

# Training
num_epochs = 200

for epoch in range(num_epochs):
    for n_batch, (real_batch,_) in enumerate (data_loader):
        N = real_batch.size(0)

        # 1. Train Disciminator
        real_data = Variable(images_to_vectors(real_batch))

        # genrate fake data and detach, so gradients are not calculated for generator()
        fake_data = generator(noise(N)).detach()

        # Train D
        d_error, d_pred_real, d_pred_fake = train_disciminator(d_optimizer,real_data, fake_data)

        # 2. Train Generator

        # genrate fake data
        fake_data = generator(noise(N))

        # Train G
        g_error = train_generator(g_optimizer,fake_data)

        # Log batch error
        logger.log(d_error, g_error, epoch, n_batch, num_batches)

        # display progress every few bathces
        if (n_batch) % 100 == 0:
            test_images = vectors_to_images(generator(test_noise))
            test_images = test_images.data
            logger.log_images(
                test_images, num_test_samples,
                epoch, n_batch, num_batches
            );
            # Display status Logs
            logger.display_status(
                epoch, num_epochs, n_batch, num_batches,
                d_error, g_error, d_pred_real, d_pred_fake
            )












# eop
