from moviepy import ImageSequenceClip
import os
# List of your images


def frames_to_vid(folder_path, fps):
    image_files = os.listdir(folder_path)
    image_files = [os.path.join(folder_path, i) for i in image_files]

    print(image_files)


    # Create video clip from images
    clip = ImageSequenceClip(image_files, fps=fps)

    # Write the output video with VP9 codec and transparency support
    clip.write_videofile(
        "output.mp4",
        codec="libx264",
        fps=fps,
        preset="ultrafast",
        ffmpeg_params=["-pix_fmt", "yuva420p"],
    )
    
if __name__ == "__main__":
    frames_to_vid("output_frames", 30)
