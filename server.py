import os
import sys
import argparse
from torch import nn
from fedlab.core.network import DistNetwork
from fedlab.contrib.algorithm.basic_server import AsyncServerHandler
from fedlab.core.server.manager import AsynchronousServerManager
from fedlab.models import MLP
import torch
from torchvision import datasets, models, transforms
import torchvision
import torchvision.transforms as transforms
import random
import torch
from torchvision import datasets, models, transforms
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split

def load_dataset():
    # Suas duas strings
    string1 = "biglycan"
    string2 = "breakhis"

    # Escolha aleatoriamente entre as duas strings
    name = random.choice([string1, string2])

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

def evaluate_model(val_loader, model):
    with torch.no_grad():
        model.eval()

        for inputs, labels in val_loader:
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)  # Supondo classificação binária
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = correct / total
    print(f'Acurácia no conjunto de validação: {100 * accuracy:.2f}%')

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
        model_ft.classifier[1] = nn.Conv2d(512, 4, kernel_size=(1,1), stride=(1,1))
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

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Distbelief training example')

    parser.add_argument('--ip', type=str, default='127.0.0.1')
    parser.add_argument('--port', type=str, default='3002')
    parser.add_argument('--world_size', type=int)
    args = parser.parse_args()

    #model = MLP(784, 10)
    model = select_model('ShuffleNet')

    handler = AsyncServerHandler(model, global_round=5)
    handler.setup_optim(0.5)

    network = DistNetwork(address=(args.ip, args.port),
                          world_size=args.world_size,
                          rank=0)
    Manager = AsynchronousServerManager(handler=handler, network=network)

    Manager.run()

    train_loader, val_loader = load_dataset()
    evaluate_model(val_loader, model)