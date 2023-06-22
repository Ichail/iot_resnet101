import torch
import os
from PIL import Image
from torchvision import transforms

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

input_transform = transforms.Compose(
    [transforms.Resize((224, 224)),
     transforms.ToTensor(),
     transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])]
)

model_path = input("Enter path to .pt model: ")
model = torch.jit.load(model_path)
model.to(device)
model.eval()
print("Model ready to work!", end="\n")
image_dir = input("Enter path to dir with image for testing .pt model: ")

for image in os.listdir(image_dir):
    full_path = os.path.join(image_dir, image)
    image = Image.open(full_path).convert('RGB')
    input_tensor = input_transform(image).to(device)
    input_batch = input_tensor.unsqueeze(0)

    with torch.no_grad():
        output = model(input_batch)

    _, predicted_idx = torch.max(output, 1)
    with open('../models/classes.txt') as f:
        classes = [line.strip() for line in f.readlines()]
    print(classes[predicted_idx.item()])
