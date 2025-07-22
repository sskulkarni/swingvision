import torch
import torchvision
import torchvision.transforms as transforms
from torchvision.models.video import r3d_18
import cv2
import numpy as np
from collections import deque

# Load pretrained I3D model (R3D-18 here as a lighter example)
model = r3d_18(pretrained=True)
model.eval()

# Class labels from Kinetics-400
KINETICS_LABELS_URL = "https://raw.githubusercontent.com/deepmind/kinetics-i3d/master/data/label_map.txt"
import urllib.request
labels = urllib.request.urlopen(KINETICS_LABELS_URL).read().decode().splitlines()

# Transform for I3D input
transform = transforms.Compose([
    transforms.Resize((112, 112)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.43216, 0.394666, 0.37645],
                         std=[0.22803, 0.22145, 0.216989])
])

# Read video and extract clips (16 frames)
def extract_clips(video_path, clip_len=16):
    cap = cv2.VideoCapture(video_path)
    frames = deque(maxlen=clip_len)
    clips = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = transform(Image.fromarray(frame))
        frames.append(frame)
        if len(frames) == clip_len:
            clip = torch.stack(list(frames))  # (T, C, H, W)
            clips.append(clip)
    cap.release()
    return clips

from PIL import Image

# Predict top action from clip
def predict_action(clip):
    clip = clip.permute(1, 0, 2, 3).unsqueeze(0)  # (1, C, T, H, W)
    with torch.no_grad():
        output = model(clip)
        probs = torch.nn.functional.softmax(output[0], dim=0)
        top5 = torch.topk(probs, k=5)
    for i in range(5):
        print(f"{labels[top5.indices[i]]}: {top5.values[i].item()*100:.2f}%")

# Example usage
#cap = cv2.VideoCapture("VideoInput/video_input4.mp4")

video_path = "VideoInput/video_input4.mp4"  # Replace with your path
clips = extract_clips(video_path)
print(f"Extracted {len(clips)} clips")

for i, clip in enumerate(clips[:6]):  # Just predict for first 3 clips
    print(f"\nClip {i+1} predictions:")
    predict_action(clip)
