import os
import subprocess
from concurrent.futures import ProcessPoolExecutor

# Input and output directories
input_dir = "/mnt/d/University/university_notes/summer_semester_24/Thesis/videos/capture/capture_10_02_2025/Ameer"
output_dir = "/mnt/d/University/university_notes/summer_semester_24/Thesis/videos/capture/capture_10_02_2025/Ameer/videos"

# Ensure output directory exists
os.makedirs(output_dir, exist_ok=True)

# Frame rate
frame_rate = "29.97002997"

# Orientations for each file
ORIENTATIONS = ["-90", "-90", "-90", "-90", "-90", "-90", "-90", "-90", "-90", "-90", "-90", "-90", "90", "90", "90", "0", "0"]

def process_video(file_path, orientation):
    """Processes a single video file using ffmpeg."""
    base_name = os.path.basename(file_path).rsplit(".", 1)[0]
    output_file = os.path.join(output_dir, f"{base_name}.mp4")

    # Determine transpose filter based on orientation
    transpose_filter = ""
    if orientation == "90":
        transpose_filter = "transpose=1"
    elif orientation == "-90":
        transpose_filter = "transpose=2"
    elif orientation == "180":
        transpose_filter = "hflip,vflip"

    # Construct ffmpeg command
    cmd = [
        "ffmpeg", "-i", file_path, "-r", frame_rate, 
        "-c:v", "libx264", "-preset", "fast", "-crf", "18", "-c:a", "aac", "-y"
    ]
    
    if transpose_filter:
        cmd.extend(["-vf", transpose_filter])

    cmd.append(output_file)

    # Execute ffmpeg command
    subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    print(f"Processed: {output_file}")

if __name__ == "__main__":
    # Get all .MOV files in input directory
    video_files = sorted([os.path.join(input_dir, f) for f in os.listdir(input_dir) if f.endswith(".MOV")])

    # Assign orientations in a cyclic manner if there are more videos than orientations
    orientation_list = [ORIENTATIONS[i % len(ORIENTATIONS)] for i in range(len(video_files))]

    # Use multiprocessing to process videos in parallel
    with ProcessPoolExecutor() as executor:
        executor.map(process_video, video_files, orientation_list)

    print("All videos processed successfully!")
