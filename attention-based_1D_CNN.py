import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
import torch
import torch.nn as nn
import torch.nn.functional as F

# Step 1: Load Video and Initialize Optical Flow
def load_video(path):
    video = cv2.VideoCapture(path)
    frames = []
    while True:
        ret, frame = video.read()
        if not ret:
            break
        frames.append(frame)
    video.release()
    return np.array(frames)

# Step 2: Calculate Optical Flow to Detect Movement
def calculate_optical_flow(frames):
    flow_magnitudes = []
    
    # Convert first frame to grayscale
    prev_gray = cv2.cvtColor(frames[0], cv2.COLOR_BGR2GRAY)
    
    for i in range(1, len(frames)):
        # Convert the next frame to grayscale
        next_gray = cv2.cvtColor(frames[i], cv2.COLOR_BGR2GRAY)
        # Calculate optical flow (Farneback method)
        flow = cv2.calcOpticalFlowFarneback(prev_gray, next_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        # Compute the magnitude of the flow vectors
        magnitude, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        # Store the average magnitude of the flow for this frame
        flow_magnitudes.append(np.mean(magnitude))
        # Update previous frame
        prev_gray = next_gray
    
    return np.array(flow_magnitudes)

# Step 3: Add Attention Layer
class TemporalAttention(nn.Module):
    def __init__(self, input_dim):
        super(TemporalAttention, self).__init__()
        self.query_layer = nn.Linear(input_dim, input_dim)
        self.key_layer = nn.Linear(input_dim, input_dim)
        self.value_layer = nn.Linear(input_dim, input_dim)
    
    def forward(self, x):
        # Compute query, key, and value
        query = self.query_layer(x)
        key = self.key_layer(x)
        value = self.value_layer(x)
        
        # Compute attention scores (scaled dot-product attention)
        attention_scores = torch.matmul(query, key.transpose(-2, -1)) / torch.sqrt(torch.tensor(query.size(-1), dtype=torch.float32))
        attention_weights = F.softmax(attention_scores, dim=-1)
        
        # Apply attention weights to the values
        output = torch.matmul(attention_weights, value)
        return output, attention_weights

# Step 4: Detect Pumping Events Using Peak Detection
def detect_pumping_rate(flow_magnitudes, threshold=0.5):
    # Find peaks in the flow magnitudes, which represent pumping events
    peaks, _ = find_peaks(flow_magnitudes, height=threshold, distance=5)
    # Calculate the pumping rate (events per second)
    pumping_rate = len(peaks) / 10  # events per second
    
    return pumping_rate, peaks

# Step 5: Visualize Attention, the Flow Magnitudes and Detected Events
def visualize_attention(attended_output, peaks):
    plt.plot(attended_output.detach().numpy(), label='Attended Flow Magnitude')
    plt.plot(peaks, attended_output[peaks].detach().numpy(), 'rx', label='Detected Events')
    plt.title('Pumping Event Detection with Attention')
    plt.xlabel('Frame Index')
    plt.ylabel('Attended Optical Flow Magnitude')
    plt.legend()
    plt.show()

def visualize_pumping(flow_magnitudes, peaks):
    plt.plot(flow_magnitudes, label='Flow Magnitude')
    plt.plot(peaks, flow_magnitudes[peaks], 'rx', label='Detected Events')
    plt.title('Pumping Event Detection')
    plt.xlabel('Frame Index')
    plt.ylabel('Optical Flow Magnitude')
    plt.legend()
    plt.show()

# Step 6: Main Function to Process Video and Estimate Pumping Rate
def main(video_path):
    frames = load_video(video_path)  # Load video frames
    flow_magnitudes = calculate_optical_flow(frames)  # Calculate optical flow
    
    # Convert flow magnitudes to tensor and apply attention
    flow_magnitudes_tensor = torch.tensor(flow_magnitudes, dtype=torch.float32).unsqueeze(0)  # Add batch dimension
    attention_layer = TemporalAttention(input_dim=1)
    attended_output, _ = attention_layer(flow_magnitudes_tensor.unsqueeze(-1))  # Add feature dimension
    attended_output = attended_output.squeeze(-1).squeeze(0)  # Remove batch and feature dimensions
    
    # Detect pumping events (peaks in flow magnitudes)
    pumping_rate, peaks = detect_pumping_rate(attended_output.detach().numpy())
    
    # Print pumping rate and visualize the results
    print(f"Estimated Pumping Rate: {pumping_rate:.2f} events/s")
    # visualize_attention(attended_output, peaks)
    visualize_pumping(flow_magnitudes, peaks)

# Step 7: Test on video
video_path = 'data/Videos_head_all/Day1_260x_Video-01.avi'
main(video_path)
#True pumping rate: 2.2 events/s
#Estimated pumping rate: 6.70 events/s