from helper import createdf, trainfunc, testfunc, trainfuncwandb
from torchvision import transforms, models
from torch.utils.data import DataLoader
from dataset import CustomDataset
import torch.nn as nn
import torch.optim as optim

if __name__ == '__main__':
    disease = 'DR'
    train_label = createdf('dataset\Training_Set\RFMiD_Training_Labels.csv',disease)
    valid_label = createdf('dataset\Evaluation_Set\RFMiD_Validation_Labels.csv',disease)
    test_label = createdf('dataset\Test_Set\RFMiD_Testing_Labels.csv',disease)

    #Dataset Dir
    train_dir = 'dataset\Training_Set\Training'
    val_dir = 'dataset\Evaluation_Set\Validation'
    test_dir = 'dataset\Test_Set\Test'

    # Define transformations
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])

    # Create CustomDataset
    train_dataset = CustomDataset(train_label,train_dir,transform)
    val_dataset = CustomDataset(valid_label,val_dir,transform)
    test_dataset = CustomDataset(test_label,test_dir,transform)

    # Create DataLoader
    batch_size = 32

    train_data_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_data_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)
    test_data_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    resnet50_model = models.resnet50(pretrained=False).to('cuda')
    # for param in resnet50_model.conv1.parameters():
    #     param.requires_grad = False
    # for param in resnet50_model.bn1.parameters():
    #     param.requires_grad = False
    # for param in resnet50_model.relu.parameters():
    #     param.requires_grad = False
    # for param in resnet50_model.maxpool.parameters():
    #     param.requires_grad = False
    # for param in resnet50_model.parameters():
    #     param.requies_grad = False
    num_classes = 2
    resnet50_model.fc = nn.Linear(resnet50_model.fc.in_features, num_classes).to('cuda')

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(resnet50_model.parameters(), lr = 0.0001)
    num_epochs = 20

    trainfunc(resnet50_model,train_data_loader,val_data_loader,optimizer,criterion,num_epochs,f'{disease}_resnet50')
    #trainfuncwandb(resnet50_model,train_data_loader,val_data_loader,optimizer,criterion,num_epochs,f'{disease}_resnet50','fundus')
    testfunc(resnet50_model,test_data_loader,criterion)