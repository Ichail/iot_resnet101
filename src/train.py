import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, datasets, models
from tqdm import tqdm
from sklearn.metrics import classification_report

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

classes = ("normal", "mirai", "ddos", "arp", "os_scan")
train_dir = 'data/train'
test_dir = 'data/test'

train_transforms = transforms.Compose(
    [transforms.Resize((224, 224)),
     transforms.ToTensor(),
     transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])]
)

test_transforms = transforms.Compose(
    [transforms.Resize((224, 224)),
     transforms.ToTensor(),
     transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])])

train_data = datasets.ImageFolder(train_dir, transform=train_transforms)
test_data = datasets.ImageFolder(test_dir, transform=test_transforms)

train_loader = torch.utils.data.DataLoader(train_data, batch_size=64, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_data, batch_size=64, shuffle=True)

model = models.resnet101(pretrained=True)

for param in model.parameters():
    param.requires_grad = False

num_classes = 5
num_features = model.fc.in_features
model.fc = nn.Linear(num_features, num_classes)

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

num_epochs = 10
model.to(device)

for epoch in range(num_epochs):
    running_loss = 0.0
    for i, data in tqdm(enumerate(train_loader)):
        inputs, labels = data[0].to(device), data[1].to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    predictions = []
    true_labels = []
    model.eval()
    with torch.no_grad():
        for data in test_loader:
            inputs, labels = data[0].to(device), data[1].to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            predictions.extend(predicted.cpu().numpy())
            true_labels.extend(labels.cpu().numpy())

    print(classification_report(true_labels, predictions, target_names=classes, zero_division=1))
    with open("report_epoch_" + str(epoch), 'w') as rep:
        rep.write(classification_report(true_labels, predictions, target_names=classes, zero_division=1))
        rep.close()

    model_scripted = torch.jit.script(model)
    model_scripted.save('model.pt')