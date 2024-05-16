from torchvision import models
import torch.nn as nn
import torch
from dataset import CustomDataset
from helper import createdf, testfunc
from torchvision import transforms
from torch.utils.data import DataLoader

if __name__ =='__main__':
    resnet50_model = models.resnet50().to('cuda')
    resnet50_model.fc = nn.Linear(resnet50_model.fc.in_features, 2).to('cuda')
    resnet50_model.load_state_dict(torch.load('models\DR_resnet50.pth'))
    criterion = nn.CrossEntropyLoss()

    disease = 'MYA'
    test_label = createdf('dataset\Test_Set\RFMiD_Testing_Labels.csv',disease)
    test_dir = 'dataset\Test_Set\Test'
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])
    test_dataset = CustomDataset(test_label,test_dir,transform)
    batch_size = 32
    test_data_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    testfunc(resnet50_model,test_data_loader,criterion)