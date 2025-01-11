from moviepy import ImageSequenceClip
import os
# List of your images
folder = "output_frames"
image_files = os.listdir(folder)
image_files = [os.path.join(folder, i) for i in image_files]

print(image_files)


# Create video clip from images
clip = ImageSequenceClip(image_files, fps=30)


# Write the output video
clip.write_videofile("output_with_transparency.mov", codec="prores_ks", preset="highest", threads=4)
