import torch
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
import torch.nn as nn
from models import LinearModel, SimpleConv
from sam.sam import SAM
import torchvision.models as models
from argparse import ArgumentParser
from tqdm import tqdm

# tensorboard
logged = True
try:
    from torch.utils.tensorboard import SummaryWriter
except:
    # when tensorboard is not installed, don't log.
    logged = False


class OptMLProj:
    def __init__(self) -> None:
        self.params = self._parse()

        torch.manual_seed(self.params.seed)

        self.device = torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu')

        assert self.params.model in ['resnet18', 'simpleconv', 'simpleconvbn']
        if self.params.model == 'resnet18':
            if self.params.batchnorm:
                self.model = models.resnet18(pretrained=False).to(self.device)
            else:
                self.model = models.resnet18(pretrained=False, norm_layer=nn.Identity).to(self.device)
        elif self.params.model == 'simpleconv':
            self.model = SimpleConv().to(self.device)
        elif self.params.model == 'simpleconvbn':
            pass

        self.transform = self.normalize()

        self.trainset = torchvision.datasets.CIFAR10(
            root='./data', train=True, download=True, transform=self.transform)
        self.trainloader = torch.utils.data.DataLoader(
            self.trainset, batch_size=self.params.batch_size, shuffle=True, num_workers=2)
        self.testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                                    download=True, transform=self.transform)
        self.testloader = torch.utils.data.DataLoader(self.testset, batch_size=self.params.batch_size,
                                                      shuffle=False, num_workers=2)
        self.classes = ('plane', 'car', 'bird', 'cat',
                        'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

        self.init_optimizer()

        self.criterion = nn.CrossEntropyLoss()

        if self.params.comment == '':
            self.params.comment = 'bz_{}_seed_{}_epochs[{}]_model_{}_baseoptim[{}]_secoptim[{}]_lr_{}_norm_{}_batchnorm_{}'.format(
                self.params.batch_size, self.params.seed, self.params.epochs, self.params.model, self.params.baseoptim,
                self.params.secoptim, self.params.lr, self.params.norm_type, self.params.batchnorm
            )
        if logged:
            self.writer = SummaryWriter(comment=self.params.comment)

    def _parse(self):
        parser = ArgumentParser(description='OptML Project code for A/SAM ')
        parser.add_argument(
            '--batch_size', help='Batch size for training/testing', type=int)
        parser.add_argument('--seed', type=int,
                            help='seed for experiments', default=0)
        parser.add_argument(
            '--sigma', help='sigma of gaussian noise', type=float, default=0.01)
        parser.add_argument('--epochs', type=int, default=100)
        parser.add_argument('--comment', type=str,
                            help='training information to save in tb', default='')
        parser.add_argument('--model', type=str,
                            help='model used in the experiment')
        parser.add_argument('--baseoptim', type=str,
                            help='base optimizer type used in experiment')
        parser.add_argument('--secoptim', type=str,
                            help='secondary optimizer type used in experiment')
        parser.add_argument('--norm_type', type=str,
                            help='normalization type')
        parser.add_argument('--lr', type=float,
                            help='learning rate', default=0.1)
        parser.add_argument('--wd', type=float,
                            help='weight_decay', default=0.005)
        parser.add_argument('--rho', type=float,
                            help='rho in a/sam', default=0.05)
        parser.add_argument('--batchnorm', action='store_true',
                            help="using batchnorm enabled by default")
        parser.add_argument('--no-batchnorm', dest='batchnorm',
                            action='store_false', help="disable batchnorm")
        parser.set_defaults(batchnorm=True)

        return parser.parse_args()

    def normalize(self):
        type = self.params.norm_type
        assert type in ['normalize', 'none']
        if type == 'normalize':
            transform = transforms.Compose(
                [transforms.ToTensor(),
                 transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        elif type == 'none':
            transform = transforms.Compose(
                [transforms.ToTensor()]
            )
        return transform

    def init_optimizer(self):
        assert self.params.baseoptim in ['sgd', 'adam']
        if self.params.baseoptim == 'sgd':
            self.base_optimizer = torch.optim.SGD
        elif self.params.baseoptim == 'adam':
            self.base_optimizer = torch.optim.Adam

        assert self.params.secoptim in ['sam', 'asam', 'none']
        if self.params.secoptim == 'sam':  # TODO: hyperparams of optimizer
            self.optimizer = SAM(self.model.parameters(),
                                 self.base_optimizer, lr=self.params.lr, weight_decay=self.params.wd, rho=self.params.rho)
        elif self.params.secoptim == 'asam':
            pass
        elif self.params.secoptim == 'none':
            self.optimizer = self.base_optimizer(self.model.parameters(), lr=self.params.lr, weight_decay=self.params.wd)

    def train(self):
        print('Start training session of: ', self.params.comment)
        # loop over the dataset multiple times
        for epoch in tqdm(range(self.params.epochs)):
            running_loss = 0.0
            epoch_loss = 0.0
            self.model.train()
            for i, data in tqdm(enumerate(self.trainloader, 0), leave=True):
                # get the inputs; data is a list of [inputs, labels]
                inputs, labels = data
                inputs, labels = inputs.to(self.device), labels.to(self.device)

                if self.params.secoptim != 'none':
                    outputs = self.model(inputs)
                    # use this loss for any training statistics
                    loss = self.criterion(outputs, labels)
                    loss.backward()
                    self.optimizer.first_step(zero_grad=True)

                    # second forward-backward pass
                    outputs = self.model(inputs)
                    # make sure to do a full forward pass
                    self.criterion(outputs, labels).backward()
                    self.optimizer.second_step(zero_grad=True)

                else:
                    # zero the parameter gradients
                    self.optimizer.zero_grad()

                    # forward + backward + optimize
                    outputs = self.model(inputs)
                    loss = self.criterion(outputs, labels)
                    loss.backward()
                    self.optimizer.step()

                # print statistics
                running_loss += loss.item()
                epoch_loss += loss.item()
                if i % 2000 == 1999:    # print every 2000 mini-batches
                    print(
                        f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 2000:.3f}')
                    running_loss = 0.0
            
            
            print('Start testing...')
            correct = 0
            total = 0
            # since we're not training, we don't need to calculate the gradients for our outputs
            self.model.eval()
            with torch.no_grad():
                for data in self.testloader:
                    images, labels = data
                    images, labels = images.to(self.device), labels.to(self.device)
                    # calculate outputs by running images through the network
                    outputs = self.model(images)
                    # the class with the highest energy is what we choose as prediction
                    _, predicted = torch.max(outputs.data, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()
            print(f'Accuracy of the network on the 10000 test images: {100 * correct // total} %')
            
            if logged:
                self.writer.add_scalar(
                    'Training loss', epoch_loss/len(self.trainloader), epoch)
                self.writer.add_scalar(
                    'Testing Accuracy', correct / total , epoch)
        print('Finished Training')

    def test(self):
        # prepare to count predictions for each class
        correct_pred = {classname: 0 for classname in self.classes}
        total_pred = {classname: 0 for classname in self.classes}

        self.model.eval()
        # again no gradients needed
        with torch.no_grad():
            for data in self.testloader:
                images, labels = data
                images, labels = images.to(self.device), labels.to(self.device)
                outputs = self.model(images)
                _, predictions = torch.max(outputs, 1)
                # collect the correct predictions for each class
                for label, prediction in zip(labels, predictions):
                    if label == prediction:
                        correct_pred[self.classes[label]] += 1
                    total_pred[self.classes[label]] += 1

        # print accuracy for each class
        for classname, correct_count in correct_pred.items():
            accuracy = 100 * float(correct_count) / total_pred[classname]
            print(f'Accuracy for class: {classname:5s} is {accuracy:.1f} %')

    def save(self):
        pass


if __name__ == '__main__':
    model = OptMLProj()
    model.train()
    model.test()
