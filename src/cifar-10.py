from __future__ import print_function
from __future__ import division

import argparse
import numpy as np
from tqdm import tqdm
from PIL import Image
import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T
from torchvision.utils import save_image


# Set up experiment
parser = argparse.ArgumentParser(description='Bidirectional Autoencoder')
parser.add_argument('dataset', help='path to dataset')
parser.add_argument('checkpoint', help='path to checkpoint or saved model')
parser.add_argument('--train', action='store_true', 
                    help='run training routine (default: False (run inference))')
parser.add_argument('--batch-size', type=int, default=1,
                    help='input batch size for validation (default: 1)')
<<<<<<< HEAD
parser.add_argument('--num-batches', type=int, 
                    help='number of testset batches to use for inference')
=======
parser.add_argument('--batches', type=int, 
                    help='number of batches to use for validation')
parser.add_argument('--save-path', default='./checkpoint.pth', 
                    help='path to temporary checkpoint/saved model (default: ./checkpoint.pth)')
>>>>>>> 5306439e5e20eae16d66c454555b4b1dde4f800e
parser.add_argument('--learn-rate', type=float, default=1e-3, 
                    help='learning rate for optimizer (default: 1e-3)')
parser.add_argument('--epochs', type=int, default=10, 
                    help='number of epochs (default: 10)')
parser.add_argument('--log-interval', type=int, default=1000, 
                    help='steps to wait before logging training status (default: 1000)')
parser.add_argument('--seed', type=int, default=7, help='random seed (default: 7)')
args = parser.parse_args()

torch.manual_seed(args.seed)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DIM = 32 * 32 * 3  # CIFAR-10 image dimension

class Autoencoder(nn.Module):
    """Defines an autoencoder network.

       Network is a 21-layer MLP with 
       identical input and layer dimensions.
    """
    def __init__(self):
        super(Autoencoder, self).__init__()
        self.fc1 = nn.Linear(DIM, DIM)
        self.fc2 = nn.Linear(DIM, DIM)
        self.fc3 = nn.Linear(DIM, DIM)
        self.fc4 = nn.Linear(DIM, DIM)
        self.fc5 = nn.Linear(DIM, DIM)
        self.fc6 = nn.Linear(DIM, DIM)
        self.fc7 = nn.Linear(DIM, DIM)
        self.fc8 = nn.Linear(DIM, DIM)
        self.fc9 = nn.Linear(DIM, DIM)
        self.fc10 = nn.Linear(DIM, DIM)
        self.fc11 = nn.Linear(DIM, DIM)
        self.fc12 = nn.Linear(DIM, DIM)
        self.fc13 = nn.Linear(DIM, DIM)
        self.fc14 = nn.Linear(DIM, DIM)
        self.fc15 = nn.Linear(DIM, DIM)
        self.fc16 = nn.Linear(DIM, DIM)
        self.fc17 = nn.Linear(DIM, DIM)
        self.fc18 = nn.Linear(DIM, DIM)
        self.fc19 = nn.Linear(DIM, DIM)
        self.fc20 = nn.Linear(DIM, DIM)
        self.fc21 = nn.Linear(DIM, DIM)

    def forward(self, x):
        x = x.view(-1, DIM)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        x = F.relu(self.fc5(x))
        x = F.relu(self.fc6(x))
        x = F.relu(self.fc7(x))        
        x = F.relu(self.fc8(x))
        x = F.relu(self.fc9(x))
        x = F.relu(self.fc10(x))
        x = F.relu(self.fc11(x))
        x = F.relu(self.fc12(x))
        x = F.relu(self.fc13(x))
        x = F.relu(self.fc14(x))
        x = F.relu(self.fc15(x))
        x = F.relu(self.fc16(x))
        x = F.relu(self.fc17(x))
        x = F.relu(self.fc18(x))
        x = F.relu(self.fc19(x))
        x = F.relu(self.fc20(x))
        x = F.relu(self.fc21(x))
        x = x.view(args.batch_size, 3, 32, 32)
        return x

# Initialize and transfer model to GPU
model = Autoencoder().to(device)

class DataIterator:
    """Wrapper for DataLoader class.
       
       Provides custom iteration over dataset.
       Future update: subclass DataLoader.
    """
    def __init__(self, dataset):
        self.dataset = dataset
        self.classes = ('plane', 'car', 'bird', 'cat', 'deer',
                        'dog', 'frog', 'horse', 'ship', 'truck')
        self.loader = torch.utils.data.DataLoader(self.dataset,
                                                 batch_size=args.batch_size,
                                                 shuffle=True, num_workers=2)

    def get_iterator(self):
        self.loader_iter = iter(self.loader)

    def release_train(self):
        """Fetches a pair of same-label tensors for training."""
        matched = False
        try:
            input1, label1 = self.loader_iter.next()
            while not matched:
                input2, label2 = self.loader_iter.next()
                if label1 == label2:
                    return input1, input2
        except StopIteration:
            return matched, matched

    def release_test(self):
        """Fetches a batch of tensors for validation."""
        images, labels = self.loader_iter.next()
        return images, labels

def cycle(x, y):
    """Runs single training step to reconstruct y from x."""
    optimizer.zero_grad()
    out = model(x.to(device))
    loss = criterion(out, y.to(device))
    loss.backward()
    optimizer.step()
    return loss.item()

def train(trainset, epochs=args.epochs, log_int=args.log_interval):
    """Routine to reconstruct image from a class counterpart."""
    dataiter = DataIterator(trainset)
    for epoch in range(epochs):
        step = 0
        running_loss = 0.0
        dataiter.get_iterator()
        
        while True:
            yin, yang = dataiter.release_train()
            if yin is False:
                break
            running_loss += cycle(yin, yang)
            running_loss += cycle(yang, yin)
            step += 2
    
            if step % log_int == 0:
                print('[Epoch %d of %d | Step %d | Loss: %.5f]' % 
                      (epoch + 1, epochs, step, running_loss/step))
                running_loss = 0.0
        
        # Save temporary checkpoint
        torch.save(model.state_dict(), args.checkpoint)

def validate(testset):
    """Perfroms inference using learned parameters."""
    params = torch.load(args.checkpoint, map_location="cpu")
    model.load_state_dict(params)

    dataiter = DataIterator(testset)
    dataiter.get_iterator()

    for batch in tqdm(range(args.num_batches)):
        images, labels = dataiter.release_test()
        name = (dataiter.classes[l.data] for l in labels)
        name = '-'.join(name)
        with torch.no_grad():
            output = model(images.to(device))
            output = output * 0.5 + 0.5
            output = output.cpu()
            save_image(output, name + '.png')

# Define data transformation
transform = T.Compose([T.ToTensor(),
                       T.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])

# Retrieve and transfom dataset
dataset = torchvision.datasets.CIFAR10(root=args.dataset, train=args.train,
                                        download=True, transform=transform)

if args.train:
    # Define loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=args.learn_rate)

    # Train the network
    train(dataset)
    print('Training complete. Model saved to: %s.' % (args.checkpoint))
else:
    # Perform inference
    validate(dataset)
