import torch
import torch.nn as nn
import cv2
import numpy as np
import matplotlib.pyplot as plt

# Step 1: Define Spatial Attention Layer
class SpatialAttention(nn.Module):
    def __init__(self, in_channels):
        super(SpatialAttention, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, 1, kernel_size=3, padding=1)  # Reduce to a single attention map
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        # x: (batch_size, channels, height, width)
        x = x.permute(1, 0, 2, 3)
        attention_map = self.sigmoid(self.conv1(x))  # Generate spatial attention map
        return x * attention_map  # Element-wise multiplication

# Step 2: Define Attention-based 3D CNN Model
class Attention3DCNN(nn.Module):
    def __init__(self, input_shape):
        super(Attention3DCNN, self).__init__()
        self.depth, self.height, self.width = input_shape[0], input_shape[1], input_shape[2]
        # 3D Convolutional Backbone
        self.conv3d_1 = nn.Conv3d(1, 16, kernel_size=(3, 3, 3), stride=(1, 2, 2), padding=(1, 1, 1))
        self.relu = nn.ReLU()
        self.pool3d = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))
        
        # Spatial Attention Block
        self.spatial_attention = SpatialAttention(16)  # Input channels = number of features
        
        # Second 3D Convolutional Layer
        self.conv3d_2 = nn.Conv3d(16, 32, kernel_size=(3, 3, 3), stride=(1, 2, 2), padding=(1, 1, 1))
        
        # Fully Connected Layer for Pumping Rate Prediction
        self.fc = nn.Linear(1476, 1)
    
    def forward(self, x):
        # 3D Convolution and Spatial Attention
        x = self.conv3d_1(x)  # First 3D convolution
        x = self.relu(x)
        x = self.pool3d(x)  # Reduce dimensions
        
        # Apply spatial attention block (frame by frame)
        x = self.spatial_attention(x)
        
        x = x.permute(1, 0, 2, 3)
        x = self.conv3d_2(x)  # Second 3D convolution
        x = self.relu(x)
        x = self.pool3d(x)  # Reduce dimensions further
        
        # Flatten and predict pumping rate
        x = x.view(x.size(0), -1)  # Flatten feature map
        x = self.fc(x)  # Fully connected layer
        return x
    
# Step 3: Preprocessing
def load_video(path):
    video = cv2.VideoCapture(path)
    frames = []
    while True:
        ret, frame = video.read()
        if not ret:
            break
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frame_resized = cv2.resize(gray_frame, (64, 48))
        frames.append(frame_resized)
    video.release()
    return np.array(frames) # frame shape = (height, width, channels)

# Step 4: Main Function to Process Video and Estimate Pumping Rate
def main(video_path):
    torch.manual_seed(0)
    video_data = load_video(video_path)  # Load video frames
    print(f"Video data shape: {video_data.shape}")
    model = Attention3DCNN(video_data.shape)
    video_tensor = torch.tensor(video_data, dtype=torch.float32).unsqueeze(0)
    model.eval()
    with torch.no_grad():
        output = model(video_tensor)  # Output shape: (batch_size, 1)
        print("Predicted pumping rate:", output.squeeze())
    
    return output.squeeze().numpy()

# Step 5: Test on Video
video_path = 'data/Videos_head_all/Day1_260x_Video-01.avi'
pumping_rate = main(video_path)
print(f"Estimated pumping rate: {np.sum(pumping_rate)/10:.2f} events/s")
# Visualize the Pumping Rates
plt.plot(pumping_rate, label="Predicted Pumping Rate")
plt.xlabel("Time step/frame")
plt.ylabel("Pumping Rate")
plt.title("Predicted Pumping Rate Over Time")
plt.legend()
plt.show()
#True pumping rate: 2.2 events/s
#Estimated pumping rate: 16.15 events/s
#Result not consistent across different random seeds