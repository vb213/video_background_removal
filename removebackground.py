from rembg import remove
import cv2
import numpy as np
from tqdm import tqdm
import sys

inputpath = sys.argv[1]
# Open the video file
cap = cv2.VideoCapture(inputpath)

# Define output video writer
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))
out = cv2.VideoWriter('output_video.mp4', cv2.VideoWriter_fourcc(*'mp4v'), fps, (frame_width, frame_height))

total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
progress_bar = tqdm(total=total_frames, desc="Processing Frames", unit="frame")


while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Process each frame to remove background
    try:
        output_frame = remove(frame)  # `remove` returns a PIL image
        output_frame = np.array(output_frame)  # Convert to NumPy array
    except Exception as e:
        print(f"Error processing frame: {e}")
        output_frame = frame  # Fallback to original frame

    # Write processed frame to output video
    out.write(output_frame)
    #cv2.imshow('Output', output_frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    progress_bar.update(1)

cap.release()
out.release()
cv2.destroyAllWindows()
