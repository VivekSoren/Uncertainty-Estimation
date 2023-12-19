import torch 
import torch.nn as nn
from torch.autograd import Variable
import torchvision.models as models

class MyDropout(nn.Module):
    def __init__(self, p=0.5):
        super(MyDropout, self).__init__()
        self.p = p
        if self.p < 1:
            self.multiplier_ = 1.0 / (1.0-p)
        else:
            self.multiplier_ = 0.0

    def forward(self, input):
        if not self.training:
            return input

        selected_ = torch.Tensor(input.shape).uniform_(0,1)>self.p
        '''
        if input.is_cuda:
            selected_ = Variable(selected_.type(torch.cuda.FloatTensor), requires_grad=False)
        else:
            selected_ = Variable(selected_.type(torch.FloatTensor), requires_grad=False)
        '''
        selected_ = Variable(selected_.type(torch.cuda.FloatTensor), requires_grad=False)

        return torch.mul(selected_, input) * self.multiplier_

class MLP(nn.Module):
    def __init__(self, hidden_layers=[800, 800], droprates=[0, 0], n_classes=14):
        super(MLP, self).__init__()
        self.model = nn.Sequential()
        # self.model.add_module("dropout0", MyDropout(p=droprates[0]))
        self.model.add_module("dropout0", nn.Dropout(p=droprates[0]))
        self.model.add_module("input", nn.Linear(28*28, hidden_layers[0]))
        self.model.add_module("tanh", nn.Tanh())

        # Add hidden layers
        for i, d in enumerate(hidden_layers[:-1]):
            # self.model.add_module("dropout_hidden"+str(i+1), MyDropout(p=droprates[1]))
            self.model.add_module("dropout_hidden"+str(i+1), nn.Dropout(p=droprates[1]))
            self.model.add_module("hidden"+str(i+1), nn.Linear(hidden_layers[i], hidden_layers[i+1]))
            self.model.add_module("tanh_hidden"+str(i+1), nn.Tanh())
        self.model.add_module("final", nn.Linear(hidden_layers[-1], n_classes))

    def forward(self, x):
        x = x.view(x.shape[0], 28*28)
        x = self.model(x)
        return x

# class MLPClassifier:
#     def __init__(self, hidden_layers=[800, 800], droprates=[0, 0]):
        self.model.cuda()

class ResNet50(nn.Module):
    def __init__(self, n_classes):
        super(ResNet50, self).__init__()
        resnet50 = models.resnet50(pretrained=True)
        self.features = nn.Sequential(*list(resnet50.children())[:-1])
        self.fc = nn.Linear(resnet50.fc.in_features, n_classes)

    def forward(self, x):
        x = self.features(x)
        # GAP layer 
        x = x.mean([2, 3])
        # Forawrd pass
        x = self.fc(x)

        return x
