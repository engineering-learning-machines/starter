import torch
from torchvision import datasets, transforms
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

RANDOM_SEED = 123456789
BATCH_SIZE = 32
VECTOR_SIZE = 784
LEARN_RATE = 0.001
LOG_INTERVAL = 10

torch.manual_seed(RANDOM_SEED)

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.hidden = nn.Linear(784, 100)
        self.output = nn.Linear(100, 10)

    def forward(self, x):
        x = F.sigmoid(self.hidden(x))
        return F.sigmoid(self.output(x))

if __name__ == '__main__':
    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST(
            '../data',
            train=True,
            download=True,
            transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,))
            ])
        ),
        batch_size=BATCH_SIZE,
        shuffle=True
    )

    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST(
            '../data',
            train=False,
            transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,))
            ])
        )
    )

    model = Net()
    optimizer = optim.SGD(model.parameters(), lr=LEARN_RATE, momentum=0.5)

    # Train
    for batch_id, (data, target) in enumerate(train_loader):
        batch_size = data.shape[0]
        data_var = Variable(data.view(batch_size, VECTOR_SIZE))
        target_var = Variable(torch.zeros(batch_size, 10).scatter_(1, target.view(batch_size, 1), 1.0))
        #
        optimizer.zero_grad()
        output = model(data_var)
        loss = nn.MSELoss()(output, target_var)
        loss.backward()
        optimizer.step()

        if batch_id % LOG_INTERVAL == 0:
            pass
            #print('Loss: {:.6f}'.format(loss.data[0]))

    # Test
    model.eval()
    test_loss = 0
    correct = 0
    for data, target in test_loader:
        #batch_size = data.shape[0]
        #data_var, target_var = Variable(data, volatile=True), Variable(target)
        data_var = Variable(data.view(VECTOR_SIZE))
        # target_var = Variable(torch.zeros(batch_size, 10).scatter_(1, target.view(batch_size, 1), 1.0))
        output = model(data_var)

        maxval, max_id = torch.max(output, 0)
        if (max_id.data[0] == target[0]):
            correct += 1

        pass

        #test_loss += nn.MSELoss()(output, target).data[0]
    print('Correct: {}/{}'.format(correct, test_loader.dataset.test_data.shape[0]))
