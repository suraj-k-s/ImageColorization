from django.shortcuts import render
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
import torch
import base64
import numpy as np
from io import BytesIO

class autoencoder_model(nn.Module):
    def __init__(self):
        super(autoencoder_model, self).__init__()

        # Autoencoder architecture
        self.down1 = nn.Conv2d(1, 64, 3, stride=2)
        self.down2 = nn.Conv2d(64, 128, 3, stride=2, padding=1)
        self.down3 = nn.Conv2d(128, 256, 3, stride=2, padding=1)
        self.down4 = nn.Conv2d(256, 512, 3, stride=2, padding=1)

        self.up1 = nn.ConvTranspose2d(512, 256, 3, stride=2, padding=1)
        self.up2 = nn.ConvTranspose2d(512, 128, 3, stride=2, padding=1)
        self.up3 = nn.ConvTranspose2d(256, 64, 3, stride=2, padding=1, output_padding=1)
        self.up4 = nn.ConvTranspose2d(128, 3, 3, stride=2, output_padding=1)

        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        d1 = self.relu(self.down1(x))
        d2 = self.relu(self.down2(d1))
        d3 = self.relu(self.down3(d2))
        d4 = self.relu(self.down4(d3))

        u1 = self.relu(self.up1(d4))
        u2 = self.relu(self.up2(torch.cat((u1, d3), dim=1)))
        u3 = self.relu(self.up3(torch.cat((u2, d2), dim=1)))
        u4 = self.sigmoid(self.up4(torch.cat((u3, d1), dim=1)))

        return u4

# Define your view
def index(request):
    if request.method == 'POST' and 'image' in request.FILES:
        image_file = request.FILES['image']
        input_gray = Image.open(image_file)

        output_img, input_gray = predict_img(input_gray)

        # Convert the tensors to numpy arrays
        output_img_numpy = output_img.squeeze().permute(1, 2, 0).numpy()
        input_gray_numpy = input_gray.squeeze().numpy()

        
        # Save the plots to separate buffers
        buffer1 = BytesIO()
        plt.imshow(input_gray_numpy, cmap='gray')
        plt.axis('off')
        plt.savefig(buffer1, format='png')
        buffer1.seek(0)

        plt.clf()  # Clear the plot

        buffer2 = BytesIO()
        plt.imshow(output_img_numpy)
        plt.axis('off')
        plt.savefig(buffer2, format='png')
        buffer2.seek(0)

        # Convert the plots to base64 strings
        input_base64 = base64.b64encode(buffer1.getvalue()).decode()
        output_base64 = base64.b64encode(buffer2.getvalue()).decode()

        return render(request, 'index.html', {'input_base64': input_base64, 'output_base64': output_base64})
    else:
        return render(request, 'index.html')


# Define your predict_img function
def predict_img(input_gray):
    # Load the saved model
    model = autoencoder_model()
    model.load_state_dict(torch.load('models/12_new_model_various_pics.pth'))

    model.eval()

    transform = transforms.Compose([
        transforms.Resize((150, 150)),
        transforms.ToTensor(),
    ])

    input_gray = input_gray.convert('L')
    input_gray = transform(input_gray).unsqueeze(0)  # Add batch dimension

    # Make prediction
    with torch.no_grad():
        predicted_color = model(input_gray)

    return predicted_color, input_gray
