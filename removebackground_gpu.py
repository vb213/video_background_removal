import cv2
import torch
import numpy as np
from tqdm import tqdm
from torchvision import transforms
from PIL import Image
import sys
import os
from u2net_model import U2NET_full


# Get input path
input_path = sys.argv[1]
background_path = "background.png"

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((320, 320)),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def insert_background_frame(frame, background, model):
    frame_height, frame_width, _ = frame.shape

    input_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    input_tensor = transform(input_image).unsqueeze(0).to(device)

    with torch.no_grad():
        prediction = model(input_tensor)[0]

    mask = prediction.squeeze().cpu().numpy()
    mask = cv2.GaussianBlur((mask > 0.5).astype(np.uint8), (5, 5), 0)  # Smooth mask
    mask = cv2.resize(mask, (frame_width, frame_height))

    # Apply the mask to create transparency
    alpha_channel = (mask * 255).astype(np.uint8)  # Convert mask to 8-bit alpha channel

    # Ensure the background has 3 channels (RGB)
    if background.shape[2] == 4:  # If background has an alpha channel, drop it
        background = background[:, :, :3]

    # Normalize the alpha channel to [0, 1]
    alpha = alpha_channel.astype(np.float32) / 255
    alpha = np.expand_dims(alpha, axis=2)  # Add a channel dimension to alpha

    # Blend the frame with the background using the alpha channel
    output_frame = (frame.astype(np.float32) * alpha + background.astype(np.float32) * (1 - alpha)).astype(np.uint8)
    return output_frame




def load_model():
    # Load U2NET model
    model_path = "saved_models/u2net.pth"
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at {model_path}")

    model = U2NET_full()
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.to(device)
    model.eval()
    return model

    # Create an output video writer (supports transparency if format allows)
    

def insert_background(input_path, background_path, output_path="output.mp4"):
    print(f"Using device: {device}")
    model = load_model()
    background = cv2.imread(background_path)

    
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise Exception(f"Failed to open video file: {input_path}")


    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    # Resize the background to match the frame dimensions
    background = cv2.resize(background, (frame_width, frame_height))

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # WebM with alpha channel (transparency support)
    out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height), isColor=True)
    if not out.isOpened():
        raise Exception("Failed to initialize the video writer.")
    
    
    # Progress bar
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    progress_bar = tqdm(total=total_frames, desc="Processing Frames", unit="frame")


    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        output_frame = insert_background_frame(frame, background, model)
        out.write(output_frame)

        # Update the progress bar
        progress_bar.update(1)


    cap.release()
    out.release()
    progress_bar.close()



insert_background(input_path, background_path)