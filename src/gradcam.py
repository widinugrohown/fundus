from matplotlib import pyplot as plt
import numpy as np
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget, SoftmaxOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image
from torchvision import datasets, transforms, models
import torch.nn as nn
import torch
from PIL import Image
import torchvision

from helper import createdf

if __name__ == '__main__':
    resnet50_model = models.resnet50(pretrained=False)
    num_classes = 2
    resnet50_model.fc = nn.Linear(resnet50_model.fc.in_features, num_classes)

    resnet50_model.load_state_dict(torch.load('models\DR_resnet50.pth'))
    target_layers = [resnet50_model.layer4[-1]]

    #Load image
    #disease = 'DR'
    #test_label = createdf('dataset\Test_Set\RFMiD_Testing_Labels.csv',disease)
    #print(test_label.head())

    # Load and preprocess your input image
    image_path = "D:/projects/fundus/dataset/Test_Set/Test/1.png"
    #image = torchvision.io.read_image(image_path)
    image = Image.open(image_path).convert('RGB')

    # Ensure the image is of type np.float32 and in the range [0, 1]
    image = np.array(image).astype(np.float32) / 255.0

    # Define image preprocessing
    preprocess = transforms.Compose([
        transforms.ToTensor()
    ])
    input_tensor = preprocess(image).unsqueeze(0)

    # Construct the GradCAM object
    cam = GradCAM(model=resnet50_model, target_layers=target_layers)

    # Specify the target class
    #target_class = [281]  # Assuming you want to visualize class 281

    # Generate the GradCAM heatmap
    grayscale_cam = cam(input_tensor=input_tensor)

    # Convert the heatmap to a visualization
    visualization_gradcam = show_cam_on_image(image, grayscale_cam[0], use_rgb=True)

    # Display the visualization
    
    plt.imshow(visualization_gradcam)
    plt.axis('off')
    plt.savefig('gradcam/gradcam.png')
    plt.show()
    