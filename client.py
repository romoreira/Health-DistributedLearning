import torchvision
import torchvision.transforms as transforms
import torch
import argparse
import sys
import os

from torch import nn

sys.path.append("../../")
from fedlab.core.client.manager import ActiveClientManager
from fedlab.core.network import DistNetwork
from fedlab.contrib.algorithm.basic_client import SGDClientTrainer
from fedlab.models import MLP
from fedlab.contrib.dataset.pathological_mnist import PathologicalMNIST
import torch
from torchvision import datasets, models, transforms
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split

def load_dataset(name):

    # Caminho para o diretório do conjunto de dados
    data_dir = "./dataset/"+str(name)

    # Transformações de pré-processamento que você deseja aplicar às imagens
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),  # Converte a imagem para tensor
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normaliza os valores dos pixels
    ])

    # Carregar o conjunto de dados
    dataset = datasets.ImageFolder(root=data_dir, transform=transform)

    # Dividir o conjunto de dados em treinamento e validação
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    # Criar DataLoader para treinamento e validação
    batch_size = 32
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)
    print("Train Loader: "+str(type(train_loader)))
    return train_loader, val_loader

def select_model(name):
    # ShuffleNet
    if name == 'ShuffleNet':
        model_ft = models.shufflenet_v2_x1_0(pretrained=True)
        # Congele todos os parâmetros do modelo (ou deixe-os descongelados, dependendo de sua escolha)
        for param in model_ft.parameters():
            param.requires_grad = True  # Defina como False para congelar
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs, 2)
        input_size = 224
        return model_ft

    elif name == 'AlexNet':
        ###AlexNet###
        feature_extract = True
        model_ft = models.alexnet(pretrained=True)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.classifier[6].in_features
        model_ft.classifier[6] = nn.Linear(num_ftrs, 4)
        input_size = 224
        return model_ft

    elif name == 'Resnet18':
        ###Resnet18###
        feature_extract = True
        model_ft = models.resnet18(pretrained=True)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs, 4)
        input_size = 224

    elif name == 'SqueezeNet':
        ###SqueezeNet###
        model_ft = models.squeezenet1_0(pretrained=True)
        set_parameter_requires_grad(model_ft, True)
        model_ft.classifier[1] = nn.Conv2d(512, 4, kernel_size=(1, 1), stride=(1, 1))
        model_ft.num_classes = 4
        input_size = 224
        return model_ft

    elif name == 'VGG11_b':
        ###VGG11_b###
        use_pretrained = True
        feature_extract = True
        model_ft = models.vgg11_bn(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.classifier[6].in_features
        model_ft.classifier[6] = nn.Linear(num_ftrs, 4)
        input_size = 224
        return model_ft

class AsyncTrainer(SGDClientTrainer):
    @property
    def uplink_package(self):
        return [self.model_parameters, self.round]

    def local_process(self, payload, id):
        model_parameters = payload[0]
        self.round = payload[1]
        #train_loader = self.dataset.get_dataloader(id, self.batch_size)
        train_loader = self.dataset
        self.train(model_parameters, train_loader)


parser = argparse.ArgumentParser(description='Distbelief training example')
parser.add_argument('--ip', type=str, default='127.0.0.1')
parser.add_argument('--port', type=str, default='3002')
parser.add_argument('--world_size', type=int)
parser.add_argument('--rank', type=int)
parser.add_argument('--dataset', type=str, help='Nome do diretório do dataset')
parser.add_argument("--epochs", type=int, default=2)
parser.add_argument("--lr", type=float, default=0.1)
parser.add_argument("--batch_size", type=int, default=100)
args = parser.parse_args()

if torch.cuda.is_available():
    args.cuda = True
else:
    args.cuda = False

#model = MLP(784, 10)
model = select_model('ShuffleNet')

train_loader, _ = load_dataset(args.dataset)
print("Client: "+str(args.rank)+" training with: "+str(args.dataset))

trainer = AsyncTrainer(model, cuda=args.cuda)
#dataset = PathologicalMNIST(root='./data/MNIST', path="./data/MNIST")
dataset = train_loader
trainer.setup_dataset(dataset)
trainer.setup_optim(args.epochs, args.batch_size, args.lr)

network = DistNetwork(address=(args.ip, args.port),
                      world_size=args.world_size,
                      rank=args.rank)

Manager = ActiveClientManager(trainer=trainer, network=network)
Manager.run()