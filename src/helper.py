import pandas as pd
import torch
from sklearn.metrics import f1_score, recall_score, precision_score, confusion_matrix

def createdf(path,label):
    filecsv = pd.read_csv(path)
    filecsv = filecsv[['ID',label]][(filecsv[label]==1) | (filecsv.Disease_Risk==0)]
    filecsv['ID'] = filecsv['ID'].astype(str)
    return filecsv

def trainfunc(model,train_data_loader,val_data_loader,optimizer,criterion,num_epochs,modelname):
    for epoch in range(num_epochs):
        model.train()
        for images, labels in train_data_loader:
            images, labels = images.to('cuda'), labels.to('cuda')
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

        # Validation loop
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in val_data_loader:
                images, labels = images.to('cuda'), labels.to('cuda')
                outputs = model(images)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                val_loss += criterion(outputs, labels).item()

        val_loss /= len(val_data_loader)
        val_accuracy = correct / total

        print(f'Epoch {epoch+1}/{num_epochs}, Train Loss: {loss.item():.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_accuracy:.4f}')

    torch.save(model.state_dict(), f'models\{modelname}.pth')
    print("Model saved successfully.")

def testfunc(model,test_data_loader,criterion):
    # Initialize lists to store true labels and predicted labels
    true_labels = []
    predicted_labels = []

    # Initialize variables for test loss and accuracy
    test_loss = 0.0
    correct = 0
    total = 0

    # Evaluation loop
    with torch.no_grad():
        for images, labels in test_data_loader:
            images, labels = images.to('cuda'), labels.to('cuda')
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            true_labels.extend(labels.cpu().numpy())
            predicted_labels.extend(predicted.cpu().numpy())

            # Update test loss and accuracy
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            test_loss += criterion(outputs, labels).item()

    # Calculate test loss and accuracy
    test_loss /= len(test_data_loader)
    test_accuracy = correct / total

    # Calculate evaluation metrics
    f1 = f1_score(true_labels, predicted_labels, average='weighted')
    recall = recall_score(true_labels, predicted_labels, average='weighted')
    precision = precision_score(true_labels, predicted_labels, average='weighted')
    conf_matrix = confusion_matrix(true_labels, predicted_labels)

    # Calculate specificity
    specificity = conf_matrix[0, 0] / (conf_matrix[0, 0] + conf_matrix[0, 1])

    print(f'Test Loss: {test_loss:.4f}, Test Acc: {test_accuracy:.4f}')
    print(f'F1 Score: {f1:.4f}, Recall: {recall:.4f}, Precision: {precision:.4f}, Specificity: {specificity:.4f}')
