import cv2
import torch
import numpy as np
from tqdm import tqdm
from torchvision import transforms
from PIL import Image
import sys
import os
from u2net_model import U2NET_full

# Get input path from command line argument
inputpath = sys.argv[1]

# Check if a GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load the U^2-Net model
model_path = "saved_models\\u2net.pth"  # Path to the pre-trained model
model = U2NET_full()
model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
model.to(device)
model.eval()

# Define image transformations
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((320, 320)),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Open the video file
cap = cv2.VideoCapture(inputpath)
if not cap.isOpened():
    raise Exception(f"Failed to open video file: {inputpath}")

frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))

# Create output directory for PNG frames
output_dir = 'output_frames'
os.makedirs(output_dir, exist_ok=True)

# Progress bar
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
progress_bar = tqdm(total=total_frames, desc="Processing Frames", unit="frame")

frame_counter = 0
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Convert the frame to a PIL Image
    input_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    # Preprocess and move to GPU
    input_tensor = transform(input_image).unsqueeze(0).to(device)

    with torch.no_grad():
        # Perform inference
        prediction = model(input_tensor)[0]  # Get the first output (saliency mask)

    # Post-process the mask
    mask = prediction.squeeze().cpu().numpy()
    mask = (mask > 0.5).astype(np.uint8)  # Binarize mask
    mask = cv2.resize(mask, (frame_width, frame_height))  # Resize mask to original frame size

    # Apply the mask to create transparency
    print("frame: ", frame)
    alpha_channel = (mask * 255).astype(np.uint8)  # Convert mask to 8-bit alpha channel
    bgr_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2BGRA)  # Add an alpha channel to the frame
    bgr_frame[:, :, 3] = alpha_channel  # Replace alpha channel with the mask

    # Save the frame as PNG with transparency
    output_filename = os.path.join(output_dir, f"frame_{frame_counter:04d}.png")
    cv2.imwrite(output_filename, bgr_frame)

    # Update the progress bar
    progress_bar.update(1)
    frame_counter += 1

cap.release()
progress_bar.close()
cv2.destroyAllWindows()
