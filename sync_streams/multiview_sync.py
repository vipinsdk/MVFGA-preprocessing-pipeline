import argparse
import os
import cv2
import numpy as np
from tqdm import tqdm
from natsort import natsorted

# from concurrent.futures import ThreadPoolExecutor
# import imutils

# INITIAL_SYNC_FRAME_INDICES = [1532, 1252, 213, 152, 335, 1418]

def time_to_frames(time_array, frame_rate=25):
    frame_indices = []
    for time in time_array:
        minutes, seconds = map(int, time.split(" "))
        total_seconds = minutes * 60 + seconds
        total_frames = int(total_seconds * frame_rate)
        frame_indices.append(total_frames)
    return frame_indices

# INITIAL_SYNC_FRAME_INDICES = ['', '', '', '', '', '', '', '', '', '', '', '', '', '', '','','']
START = [3019, 2974, 2930, 2747, 2808, 2861, 2543, 2617,2678, 2274, 2366, 2423, 2036, 2110, 2196, 2920, 2722]
EX_INITIAL_SYNC_FRAME_INDICES = ['0 23','0 33']
END_SYNC_FRAME_INDICES = None


# start synchronized frame indices: [354, 522, 241, 153]
# end synchronized frame indices: [2354, 2522, 2241, 2153]

class MultiVideoSyncer:
    def __init__(self, video_paths, sync_indices=None):
        self.video_paths = video_paths
        self.caps = [cv2.VideoCapture(path) for path in video_paths]
        
        if sync_indices is not None:
            self.current_frame_indices = sync_indices
        else:
            self.current_frame_indices = [0] * len(self.caps)
        
        self.monitor_width = 1920
        self.monitor_height = 1080

    def close_streams(self):
        for cap in self.caps:
            cap.release()

    def sync_videos(self):
        print('''
Keyboard Controls:
    W/S -> Forward/Backward Frame for All Videos
    D/A -> Forward/Backward Frame for Selected Video
    G/F -> Forward/Backward 10 Frames for Selected Video
    T/Y -> Forward/Backward 10 Frames for Selected Video
    B/V -> Forward/Backward 10 Frames for All Videos
    P/O -> Forward/Backward 100 Frames for All Videos
    H/J -> Forward/Backward 50 Frames for Selected Videos
    L/K -> Forward/Backward 100 Frames for Selected Videos
    N -> Forward 500 Frames for All Videos
    M -> Backward 500 Frames for All Videos
    I -> Forward 1000 Frames for All Videos
    U -> Backward 1000 Frames for All Videos

    Q -> Exit Sync Process
    Mouse Click -> Select Video to Adjust Individually
    ''')

        selected_video = 0
        overlayed_frame = None

        while True:
            frames = []
            for i, cap in enumerate(self.caps):
                cap.set(cv2.CAP_PROP_POS_FRAMES, self.current_frame_indices[i])
                ret, frame = cap.read()
                if not ret:
                    frame = np.zeros((self.monitor_height, self.monitor_width // len(self.caps), 3), dtype=np.uint8)  # Placeholder for empty frame

                num_videos = len(self.caps)
                cols = int(np.ceil(np.sqrt(num_videos * self.monitor_width / self.monitor_height)))
                rows = int(np.ceil(num_videos / cols))

                target_width = self.monitor_width // cols
                target_height = self.monitor_height // rows

                frame = cv2.resize(frame, (target_width, target_height))

                # Add rectangle border for the selected video
                if i == selected_video:
                    cv2.rectangle(frame, (0, 0), (frame.shape[1] - 1, frame.shape[0] - 1), (0, 255, 0), 3)

                frames.append(frame)

            # Overlay all frames into one window
            overlayed_frame = self.create_overlay(frames, cols)
            cv2.imshow("Synced Videos", overlayed_frame)

            key = cv2.waitKey(1) & 0xFF

            if key == ord('q'):
                break
            elif key == ord('w'):
                self.current_frame_indices = [idx + 1 for idx in self.current_frame_indices]
            elif key == ord('s'):
                self.current_frame_indices = [max(0, idx - 1) for idx in self.current_frame_indices]
            elif key == ord('d'):
                self.current_frame_indices[selected_video] += 1
            elif key == ord('a'):
                self.current_frame_indices[selected_video] = max(0, self.current_frame_indices[selected_video] - 1)
            elif key == ord('g'):
                self.current_frame_indices[selected_video] += 10
            elif key == ord('f'):
                self.current_frame_indices[selected_video] = max(0, self.current_frame_indices[selected_video] - 10)
            elif key == ord('t'):
                self.current_frame_indices[selected_video] += 5
            elif key == ord('r'):
                self.current_frame_indices[selected_video] = max(0, self.current_frame_indices[selected_video] - 5)
            elif key == ord('b'):
                self.current_frame_indices = [idx + 10 for idx in self.current_frame_indices]
            elif key == ord('v'):
                self.current_frame_indices = [max(0, idx - 10) for idx in self.current_frame_indices]
            elif key == ord('h'):
                self.current_frame_indices[selected_video] += 50
            elif key == ord('j'):
                self.current_frame_indices[selected_video] = max(0, self.current_frame_indices[selected_video] - 50)
            elif key == ord('l'):
                self.current_frame_indices[selected_video] += 100
            elif key == ord('k'):
                self.current_frame_indices[selected_video] = max(0, self.current_frame_indices[selected_video] - 100)        
            elif key == ord('p'):
                self.current_frame_indices = [idx + 100 for idx in self.current_frame_indices]
            elif key == ord('o'):
                self.current_frame_indices = [max(0, idx - 100) for idx in self.current_frame_indices]

            elif key == ord('n'):
                self.current_frame_indices = [idx + 500 for idx in self.current_frame_indices]
            elif key == ord('m'):
                self.current_frame_indices = [max(0, idx - 500) for idx in self.current_frame_indices]
            elif key == ord('i'):
                self.current_frame_indices = [idx + 1000 for idx in self.current_frame_indices]
            elif key == ord('u'):
                self.current_frame_indices = [max(0, idx - 1000) for idx in self.current_frame_indices]

            def select_video(event, x, y, flags, param):
                nonlocal selected_video
                if event == cv2.EVENT_LBUTTONDOWN:
                    height, width, _ = overlayed_frame.shape
                    video_width = width // cols
                    video_height = height // rows
                    col = x // video_width
                    row = y // video_height
                    selected_video = row * cols + col


            cv2.setMouseCallback("Synced Videos", select_video)

        self.close_streams()
        return self.current_frame_indices

    @staticmethod
    def create_overlay(frames, cols):
        # Arrange frames in a grid layout
        rows = int(np.ceil(len(frames) / cols))
        blank_frame = np.zeros_like(frames[0])
        grid = []
        for r in range(rows):
            row_frames = frames[r * cols:(r + 1) * cols]
            while len(row_frames) < cols:
                row_frames.append(blank_frame)
            grid.append(np.hstack(row_frames))
        return np.vstack(grid)


def main():
    parser = argparse.ArgumentParser(description='Synchronize multiple videos.')
    parser.add_argument('--video_dir', required=True, type=str, help='Directory containing videos to synchronize.')
    args = parser.parse_args()

    video_dir = args.video_dir
    video_paths = [os.path.join(video_dir, f) for f in natsorted(os.listdir(video_dir)) if f.endswith('.mp4')]
    # video_paths = [video_paths[i] for i in ext_2]

    if not video_paths:
        print("No video files found in the specified directory.")
        return

    # syncer = MultiVideoSyncer(video_paths, sync_indices=time_to_frames(INITIAL_SYNC_FRAME_INDICES[8:]))
    syncer = MultiVideoSyncer(video_paths, sync_indices=START)
    start_frame_indices = syncer.sync_videos()
    print("start synchronized frame indices:", start_frame_indices)

    if END_SYNC_FRAME_INDICES is not None:
        sync_indices = END_SYNC_FRAME_INDICES
    else:
        sync_indices = start_frame_indices

    syncer = MultiVideoSyncer(video_paths, sync_indices=sync_indices)
    end_frame_indices = syncer.sync_videos()
    print("end synchronized frame indices:", end_frame_indices)

    with open(video_dir + '/sync_streams.txt', "w") as f:
        f.write(f"START: {start_frame_indices}\n")
        f.write(f"END: {end_frame_indices}")


if __name__ == '__main__':
    main()