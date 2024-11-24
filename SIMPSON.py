import os
import torch
import torchvision
import torchvision.transforms as transforms
import torchvision.models as models
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from torchvision import datasets,transforms
from torch.utils.data import random_split
from torch.utils.data import Dataset
from torch.utils.data import ConcatDataset

print(torch.__version__)
#GPU
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
#Assuming that we are on a CUDA machine, this should print a CUDA device:
print(device)

# Custom transform to add Gaussian noise
class AddGaussianNoise(object):
    def __init__(self, mean=0., std=1.):
        self.std = std
        self.mean = mean

    def __call__(self, tensor):
        return tensor + torch.randn(tensor.size()) * self.std + self.mean

# Custom transform to add Speckle noise
class AddSpeckleNoise(object):
    def __init__(self, noise_level=0.1):
        self.noise_level = noise_level

    def __call__(self, tensor):
        # Generate speckle noise
        noise = torch.randn_like(tensor) * self.noise_level
        # Add speckle noise to the image
        noisy_tensor = tensor * (1 + noise)
        # Clip the values to be between 0 and 1
        noisy_tensor = torch.clamp(noisy_tensor, 0, 1)
        return noisy_tensor

class AddPoissonNoise(object):
    def __init__(self, lam=1.0):
        self.lam = lam

    def __call__(self, tensor):
        # Generate Poisson noise
        noise = torch.poisson(self.lam * torch.ones(tensor.shape))
        # Add Poisson noise to the image
        noisy_tensor = tensor + noise / 255.0  # Assuming the image is scaled between 0 and 1
        # Clip the values to be between 0 and 1
        noisy_tensor = torch.clamp(noisy_tensor, 0, 1)
        return noisy_tensor
    
# Custom transform to add Salt and Pepper noise
class AddSaltPepperNoise(object):
    def __init__(self, salt_prob=0.05, pepper_prob=0.05):
        self.salt_prob = salt_prob
        self.pepper_prob = pepper_prob

    def __call__(self, tensor):
        noise = torch.rand(tensor.size())
        tensor[(noise < self.salt_prob)] = 1  
        tensor[(noise > 1 - self.pepper_prob)] = 0
        return tensor

#org data
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])
#data aug
transformAug1 = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomApply([transforms.RandomHorizontalFlip()], p=0.1),
    transforms.RandomApply([transforms.RandomVerticalFlip()], p=0.1),
    transforms.RandomApply([transforms.RandomRotation(10)], p=0.1),
    transforms.RandomApply([transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1)], p=0.1),
    transforms.RandomGrayscale(p=0.1),
    transforms.RandomInvert(p=0.1),
    transforms.RandomPosterize(bits=2, p=0.1),
    transforms.RandomApply([transforms.RandomSolarize(threshold=1.0)], p=0.1),
    transforms.RandomApply([transforms.RandomAdjustSharpness(sharpness_factor=2)], p=0.1),
    transforms.RandomApply([transforms.RandomPerspective(distortion_scale=0.6, p=1.0)], p=0.1),
    transforms.RandomApply([transforms.RandomAffine(degrees=(30, 70), translate=(0.1, 0.3), scale=(0.5, 0.75))], p=0.1),
    transforms.RandomApply([transforms.ElasticTransform(alpha=250.0)], p=0.1),
    transforms.RandomApply([transforms.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 5.))], p=0.1),
    transforms.ToTensor(),
    transforms.RandomApply([AddGaussianNoise(0., 0.05)], p=0.1),  # mean and std
    transforms.RandomApply([AddSpeckleNoise(noise_level=0.1)], p=0.1),
    transforms.RandomApply([AddPoissonNoise(lam=0.1)], p=0.1),
    transforms.RandomApply([AddSaltPepperNoise(salt_prob=0.05, pepper_prob=0.05)], p=0.1),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])
transformAug2 = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomApply([transforms.RandomHorizontalFlip()], p=0.2),
    transforms.RandomApply([transforms.RandomVerticalFlip()], p=0.2),
    transforms.RandomApply([transforms.RandomRotation(10)], p=0.2),
    transforms.RandomApply([transforms.ColorJitter(brightness=0.6, contrast=0.6, saturation=0.6, hue=0.1)], p=0.2),
    transforms.RandomGrayscale(p=0.2),
    transforms.RandomInvert(p=0.2),
    transforms.RandomPosterize(bits=3, p=0.2),
    transforms.RandomApply([transforms.RandomSolarize(threshold=1.0)], p=0.2),
    transforms.RandomApply([transforms.RandomAdjustSharpness(sharpness_factor=3)], p=0.2),
    transforms.RandomApply([transforms.RandomPerspective(distortion_scale=0.7, p=1.0)], p=0.2),
    transforms.RandomApply([transforms.RandomAffine(degrees=(40, 70), translate=(0.2, 0.4), scale=(0.5, 0.8))], p=0.2),
    transforms.RandomApply([transforms.ElasticTransform(alpha=200.0)], p=0.2),
    transforms.RandomApply([transforms.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 5.))], p=0.2),
    transforms.ToTensor(),
    transforms.RandomApply([AddGaussianNoise(0., 0.1)], p=0.2),  # mean and std
    transforms.RandomApply([AddSpeckleNoise(noise_level=0.1)], p=0.2),
    transforms.RandomApply([AddPoissonNoise(lam=0.1)], p=0.5),
    transforms.RandomApply([AddSaltPepperNoise(salt_prob=0.1, pepper_prob=0.1)], p=0.2),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])

trainDataset = datasets.ImageFolder("/home/sil313505007/ML2/train/train/", transform=transform)
aug1Dataset = datasets.ImageFolder("/home/sil313505007/ML2/train/train/", transform=transformAug1)
aug2Dataset = datasets.ImageFolder("/home/sil313505007/ML2/train/train/", transform=transformAug2)

combinDataset = ConcatDataset([trainDataset, aug1Dataset, aug2Dataset])
train_size = int(0.8 * len(combinDataset))
validation_size = len(combinDataset) - train_size
trainSet, validationSet = random_split(combinDataset, [train_size, validation_size])
trainSetloader = torch.utils.data.DataLoader(trainSet, batch_size=8, shuffle=False, num_workers=2)
validationSetloader = torch.utils.data.DataLoader(validationSet, batch_size=8, shuffle=False, num_workers=2)

print(f"Origin DataSet number : {len(trainDataset)}")
print(f"After +augDataSet number : {len(trainSet)}")

testDataset = datasets.ImageFolder("/home/sil313505007/ML2/test-final", transform=transform)
testloader = torch.utils.data.DataLoader(testDataset, batch_size=8, shuffle=False, num_workers=2)
testImages = {}
for i in range(len(testDataset)):
    image, _ = testDataset[i]
    file_name = os.path.basename(testDataset.samples[i][0])
    key = os.path.splitext(file_name)[0]
    testImages[key] = image

#date info
print("Number of classes:", len(trainDataset.classes))
print("Class names:", trainDataset.classes)
print("==================================================")
trainClass_counts = {class_name: 0 for class_name in trainDataset.classes}
for img_path,class_idx in trainDataset.samples: #trainDataset
    class_name = trainDataset.classes[class_idx]
    trainClass_counts[class_name] += 1
for class_name, count in trainClass_counts.items():
    print(f"Class: {class_name}, Count: {count}")
print("==================================================")
class_counts = {class_name: 0 for class_name in testDataset.classes}
for img_path,class_idx in testDataset.samples: #testDataset
    class_name = testDataset.classes[class_idx]
    class_counts[class_name] += 1
for class_name, count in class_counts.items():
    print(f"Class: {class_name}, Count: {count}")



print("model start")
#model
model = models.efficientnet_v2_s().to(device)
# model.to(device)
num_classes = len(trainDataset.classes)
model.classifier = nn.Linear(1280, num_classes).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
num_epochs = 100
epoch_list = []
loss_list = []
# Training loop
for epoch in range(num_epochs):
    running_loss = 0.0
    correct = 0
    total = 0
    for i, data in enumerate(trainSetloader, 0):
        inputs, labels = data[0].to(device), data[1].to(device)
        # Zero the parameter gradients
        optimizer.zero_grad()
        # Forward pass
        outputs = model(inputs)
        outputs = outputs.to(device)
        loss = criterion(outputs,labels)
        # Backward pass and optimization
        loss.backward()
        optimizer.step()
        # Print statistics
        running_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    loss_avg = running_loss / len(trainSetloader)
    accuracy = 100 * correct / total
    print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {loss_avg:.3f}, Accuracy: {accuracy:.2f}%")
    epoch_list.append(epoch + 1)
    loss_list.append(loss_avg)
    if (epoch + 1) % 5 == 0:  # save every 5 epoch
        PATH = "/home/sil313505007/ML2/model_epoch_{}.4.pth".format(epoch + 1)
        torch.save(model.state_dict(), PATH) 
print("Training complete")
#save epoch
epochData = {'Epoch': epoch_list, 'Loss': loss_list}
epdf = pd.DataFrame(epochData)
csv_file_path = "/home/sil313505007/ML2/epoch_loss_data.csv"
epdf.to_csv(csv_file_path, index=False)
print("Saving epoch Data complete")

#predic
correct = 0
total = 0
model.eval()
with torch.no_grad():
    for data in validationSetloader:
        images, labels = data[0].cuda(), data[1].cuda()
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
print(f"Accuracy on the test data: {100 * correct / total}%")
# prepare to count predictions for each class
correct_pred = {classname: 0 for classname in trainDataset.classes}
total_pred = {classname: 0 for classname in trainDataset.classes}
#no gradients needed
with torch.no_grad():
    for data in validationSetloader:
        images, labels = data[0].cuda(), data[1].cuda()
        outputs = model(images)
        _, predictions = torch.max(outputs, 1)
        # collect the correct predictions for each class
        for label, prediction in zip(labels, predictions):
            if label == prediction:
                correct_pred[trainDataset.classes[label]] += 1
            total_pred[trainDataset.classes[label]] += 1
# print accuracy for each class
for classname, correct_count in correct_pred.items():
    accuracy = 100 * float(correct_count) / total_pred[classname]
    print(f'Accuracy for class: {classname:5s} is {accuracy:.1f} %')

#output
model.eval() 
for key, image in testImages.items():
    image = image.to(device)
    image = image.unsqueeze(0)
    with torch.no_grad():
        output = model(image)
    _, predicted = torch.max(output, 1)
    predictedLabel = trainDataset.classes[predicted.item()]
    testImages[key] = predictedLabel
df = pd.DataFrame({'id': list(testImages.keys()), 'character': list(testImages.values())})
df.to_csv("/home/sil313505007/ML2/output.csv", index=False)
df = pd.read_csv("/home/sil313505007/ML2/output.csv")
df = df.sort_values(by='id')
df.to_csv("/home/sil313505007/ML2/predictions.csv", index=False)
print("All complete")