import argparse
import os
import cv2
import tqdm
from natsort import natsorted

def trim_videos(video_dir, output_dir):
    """
    Trims videos based on start and end frame indices provided in the sync file.

    Args:
        video_dir (str): Directory containing the videos to trim.
        sync_file (str): Path to the sync file containing start and end frame indices.
        output_dir (str): Directory to save the trimmed videos.
    """
    sync_file = os.path.join(video_dir, 'sync_streams.txt')
    assert os.path.exists(sync_file), f"Sync file not found at {sync_file}."

    # Load start and end indices from the sync file
    with open(sync_file, 'r') as f:
        lines = f.readlines()
        start_indices = eval(lines[0].split(': ')[1].strip())
        end_indices = eval(lines[1].split(': ')[1].strip())

    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    video_paths = [os.path.join(video_dir, f) for f in natsorted(os.listdir(video_dir)) if f.endswith('.mp4')]

    if len(video_paths) != len(start_indices) or len(video_paths) != len(end_indices):
        raise ValueError("Mismatch between number of videos and sync indices.")

    for i, video_path in enumerate(tqdm.tqdm(video_paths, desc="Trim videos")):
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        start_frame = start_indices[i]
        end_frame = end_indices[i]

        if start_frame >= end_frame:
            print(f"Skipping video {os.path.basename(video_path)} due to invalid frame range.")
            cap.release()
            continue

        # Calculate output video path
        output_path = os.path.join(output_dir, os.path.basename(video_path))

        # Open a VideoWriter to save the trimmed video
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        for frame_idx in range(start_frame, end_frame):
            ret, frame = cap.read()
            if not ret:
                print(f"Warning: Unable to read frame {frame_idx} from {os.path.basename(video_path)}.")
                break
            out.write(frame)

        cap.release()
        out.release()
        print(f"Trimmed video saved to {output_path}")

def main():
    parser = argparse.ArgumentParser(description='Trim videos based on sync indices.')
    parser.add_argument('--video_dir', required=True, type=str, help='Directory containing the videos to trim.')
    parser.add_argument('--output_dir', required=True, type=str, help='Directory to save the trimmed videos.')
    args = parser.parse_args()

    trim_videos(args.video_dir, args.output_dir)

if __name__ == '__main__':
    main()
