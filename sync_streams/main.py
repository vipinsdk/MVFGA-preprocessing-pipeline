import sys; sys.path.append('..')
import argparse
import json
import pickle
import os
import cv2
import imutils
import numpy as np
import h5py
from rich import print

from tqdm import trange
from concurrent.futures import ThreadPoolExecutor
from config import TEMP_DIR
from read_captury_joint_position import parse_file


from tqdm import tqdm
from natsort import natsorted
from dv import AedatFile


OUTPUT_WIDTH = 640 
OUTPUT_HEIGHT = 480


class SyncCamera:
    def open_streams(self):
        # self.cap = cv2.VideoCapture(self.rgb_stream_path)
        self.ecap = AedatFile(self.aedat_stream_path)
        
    def close_streams(self):
        # self.cap.release()
        self.ecap.close()
        cv2.destroyAllWindows()

    def save_data(self, event_frame, event_cloud, current_frame_index):
        cloud_path = os.path.join(self.event_cloud_path, f'{current_frame_index}.npy')
        frame_path = os.path.join(self.event_frames_path, f'{current_frame_index}.jpg')

        if os.path.exists(cloud_path) and os.path.exists(frame_path):
            return

        np.save(cloud_path, np.array(event_cloud))
        cv2.imwrite(frame_path, event_frame * 255)

    def accumulate_events_by_fps(self, frame_time_ms):
        folder_name = self.ego_path.replace(os.path.sep, '_').replace(':', '_')
        self.folder_path = os.path.join(TEMP_DIR, folder_name)
        
        self.event_cloud_path = os.path.join(self.folder_path, 'event_clouds')        
        self.event_frames_path = os.path.join(self.folder_path, 'event_frames')        

        event_frame_path = os.path.join(self.folder_path, 'event_frame_map.npy')

        if os.path.exists(event_frame_path):
            print(f'Loading Event Frame Map from {event_frame_path}')
            return np.load(event_frame_path, mmap_mode='r')

        print('Creating Event Frame Map')

        os.makedirs(self.event_frames_path, exist_ok=True)
        os.makedirs(self.event_cloud_path, exist_ok=True)

        workers = ThreadPoolExecutor(max_workers=256)

        events = self.events
        
        timestamps, xs, ys, polarities = events['timestamp'], events['x'], events['y'], events['polarity']
        timestamps = (timestamps - timestamps[0]) * 1e-3
        
        event_frame = np.zeros((OUTPUT_HEIGHT, OUTPUT_WIDTH), dtype=np.uint8)
        event_cloud = list()
        event_frame_map = np.zeros((len(timestamps), 2), dtype=np.uint64)
        current_frame_index = 0 
        current_frame_time = frame_time_ms 
        for idx, ts in enumerate(tqdm(timestamps)):
            x, y, pol = xs[idx], ys[idx], polarities[idx]
            
            if ts < current_frame_time:
                event_frame[y, x] = 1
                event_cloud.append((x, y, ts, pol))

                event_frame_map[idx, 0] = idx
                event_frame_map[idx, 1] = current_frame_index   
            else:
                workers.submit(self.save_data, event_frame, event_cloud, current_frame_index)

                current_frame_index += 1
                current_frame_time += frame_time_ms
                event_frame = np.zeros((OUTPUT_HEIGHT, OUTPUT_WIDTH))
                event_cloud = list()

        print('Saving Event Frame Map')
        np.save(event_frame_path, event_frame_map)

        return event_frame_map

    def __init__(self, ego_path, caps, aedat_stream_path, pose_list):
        self.ego_path = ego_path
        self.pose_list = pose_list
        self.caps = caps  
        self.aedat_stream_path = aedat_stream_path 
        
        print('Loading Events')
        self.open_streams()

        stream_fps = self.caps[0].get(cv2.CAP_PROP_FPS)
        self.fps = 59.940060# self.caps[0].get(cv2.CAP_PROP_FPS)

        print(f'Set FPS: {self.fps}')
        print(f'[bold red]Stream FPS: {stream_fps}')
        
        self.frame_time_ms = (1 / self.fps) * 1000 # sec to ms
        
        self.events = np.hstack([packet for packet in self.ecap['events'].numpy()])
        
        self.event_frame_map = self.accumulate_events_by_fps(self.frame_time_ms)
        self.close_streams()
        print('Done Loading Events')
    
    def get_event_frame(self, index):
        frame_path = os.path.join(self.event_frames_path, f'{index}.jpg')
        
        image = cv2.imread(frame_path, cv2.IMREAD_GRAYSCALE) 

        if image is None:
            print(f'[bold red]Event frame {index} not found at {frame_path}')
            image = np.zeros((OUTPUT_HEIGHT, OUTPUT_WIDTH), dtype=np.uint8)

        return image / 255


    def get_fine_rgb_sync_frame_id(self):
        print('''W -> Forward RGB Sequence, S -> Backward RGB Sequence, D -> Forward Event Sequence, A -> Backward Event Sequence, 
                'G -> Forward 10 RGB Sequence, F -> Backward 10 RGB Sequence, B -> Forward 10 Event Sequence, V -> Backward 10 Event Sequence
                q - Set sync frame start.
                p - Forward 100 RGB Sequence, 100 Event Sequence 
                [,] to change stream
                ''')
        self.open_streams()
        
    
        caps = self.caps
        number_of_streams = len(caps)
        stream_id = 0
        
        eindex = 0
        findex = 0
        # r = self.r 
        while True:
            cap = caps[stream_id]

            cap.set(cv2.CAP_PROP_POS_FRAMES, findex)        
            ret, frame = cap.read()

            if not ret: break        
            frame = imutils.resize(frame, width=1080)
            
            eframe = self.get_event_frame(eindex)
            
            cv2.imshow('event_frame', eframe)
            cv2.imshow('ext_frame', frame)
            c = cv2.waitKey(0)

            if ord('w') == c:
                findex += 1
            elif ord('s') == c:
                findex -= 1
            elif ord('a') == c:
                eindex -= 1
            elif ord('d') == c:
                eindex += 1

            elif ord('g') == c:
                findex += 10
            elif ord('f') == c:
                findex -= 10
            elif ord('v') == c:
                eindex -= 10
            elif ord('b') == c:
                eindex += 10

            elif ord('p') == c:
                findex += 100
                eindex += 100

            if c == ord('['):
                stream_id = (stream_id + 1) % number_of_streams

            if c == ord(']'):
                stream_id = (stream_id - 1) % number_of_streams

            cap = caps[stream_id]

            if ord('q') == c:
                break
            
            print(f'ext_index: {findex} ego_index: {eindex} stream_id: {stream_id}')
        
        self.close_streams()

        return findex, eindex, stream_id

    def get_number_of_frames(self, findex, eindex, stream_id):
        print('''W -> Forward RGB Sequence, S -> Backward RGB Sequence, 
                 F -> Fast/Slow seek mode, 
                 q -> Set number of frames.
                 G -> Forward 100 Frame Sequences
                [,] to change stream
                ''')

        self.open_streams()
    
        caps = self.caps
        number_of_streams = len(caps)

        seek_mode = 'slow'
        num_frames = 0
        while True:
            cap = caps[stream_id]

            cap.set(cv2.CAP_PROP_POS_FRAMES, findex + num_frames)        
            ret, frame = cap.read()
            
            try: eframe = self.get_event_frame(eindex + num_frames)
            except IndexError: break

            if not ret: break        
            frame = imutils.resize(frame, width=1080)
            # eframe = cv2.cvtColor(cv2.resize(eframe, (frame.shape[1], frame.shape[0])), cv2.COLOR_GRAY2BGR)

            # cv2.imshow('stacked_frame', np.hstack((frame, eframe)))

            cv2.imshow('event_Frame', eframe)
            cv2.imshow('ext_frame', frame)

            if seek_mode == 'fast':
                c = cv2.waitKey(1)
            else:
                c = cv2.waitKey(0)

            if seek_mode == 'fast':
                num_frames += 1
            else:
                if ord('w') == c:
                    num_frames += 1
                elif ord('s') == c:
                    num_frames -= 1
        
            print(f'FrameIndex: ext_frame: {findex + num_frames} event_Frame: {eindex + num_frames} n_frames: {num_frames}')
            
            if ord('f') == c:                
                if seek_mode == 'fast':
                    seek_mode = 'slow'
                else: seek_mode = 'fast'
            
            if ord('g') == c:
                num_frames += 100
            
            if ord('q') == c:
                break

            if c == ord('['):
                stream_id = (stream_id + 1) % number_of_streams

            if c == ord(']'):
                stream_id = (stream_id - 1) % number_of_streams

    
        self.close_streams()

        return num_frames

    def __call__(self, args):
        stream_id = args.stream_id

        findex = args.ext_index
        eindex = args.ego_index
        n_frames = args.n_frames

        if findex is None or eindex is None:
            findex, eindex, stream_id = self.get_fine_rgb_sync_frame_id()
        
        if n_frames is None:
            n_frames = self.get_number_of_frames(findex, eindex, stream_id)
        
        pose_list = self.pose_list[findex:]
        n_frames = min(n_frames, len(pose_list) - 1)
        
        pose_list = pose_list[:n_frames]

        print(f'Sync Frame Index.  Ext Camera Index: {findex} Ego Camera Index: {eindex} Number Of frames: {n_frames} stream_id: {stream_id}')

        events = self.events
        timestamps, xs, ys, polarities = events['timestamp'], events['x'], events['y'], events['polarity']

        events = list()

        event_indices = self.event_frame_map[:, 0]
        event_frame_indices = self.event_frame_map[:, 1]
        
        index_map = (event_frame_indices >= eindex) * (event_frame_indices <= (eindex + n_frames))
        
        event_indices = event_indices[index_map]
        event_frame_indices = event_frame_indices[index_map] - eindex
        
        segmentation_indices = np.ones_like(event_frame_indices) * -1

        pickle.dump({'frame_start_index': findex, 'pose_list': pose_list},
                     open(os.path.join(self.ego_path, f'synced_pose_gt.pickle'), 'wb'))
    
        xs = xs[event_indices][..., None]
        ys = ys[event_indices][..., None]
        timestamps = timestamps[event_indices][..., None]
        polarities = polarities[event_indices][..., None]
        event_frame_indices = event_frame_indices[..., None]
        segmentation_indices = segmentation_indices[..., None]
                
        BATCH_SIZE = 8192 * 10

        print(f'Writing events to disk: {self.ego_path}/events.h5')
        h5_file = h5py.File(os.path.join(self.ego_path, 'events.h5'), 'w')

        create_dataset = True
        for i in trange(0, xs.shape[0], BATCH_SIZE):
            events = np.concatenate([
                                     xs[i:i+BATCH_SIZE], 
                                     ys[i:i+BATCH_SIZE], 
                                     timestamps[i:i+BATCH_SIZE], 
                                     polarities[i:i+BATCH_SIZE], 
                                     event_frame_indices[i:i+BATCH_SIZE], 
                                     segmentation_indices[i:i+BATCH_SIZE]
                                     ], -1)

            if create_dataset:
                h5_file.create_dataset('event', data=events, chunks=True, maxshape=(None, events.shape[1]))
                create_dataset = False
            else:
                dataset = h5_file['event']

                # Determine the shape of the existing dataset
                existing_shape = dataset.shape
                # Determine the shape of the new data
                new_shape = events.shape
                # Adjust the shape of the existing dataset to accommodate the new data
                dataset.resize(existing_shape[0] + new_shape[0], axis=0)
                # Append the new data to the existing dataset
                dataset[existing_shape[0]:] = events


        with open(os.path.join(self.ego_path, 'event_meta.json'), 'w') as fout:            
            json.dump({ 'width': OUTPUT_WIDTH, 
                        'height': OUTPUT_HEIGHT,
                        'columns': ['x', 'y', 'timestamp', 'polarity', 'frame_index', 'segmentation_indices'],
                        }, 
                        fout)

        h5_file.close()
        print('Done Writing events to disk')


def main():

    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--ego_path', required=True, type=str)    
    parser.add_argument('--ext_path', required=True, type=str)
    parser.add_argument('--stream_id', required=True, type=int)

    parser.add_argument('--ext_index', required=False, default=None, type=int)
    parser.add_argument('--ego_index', required=False, default=None, type=int)
    parser.add_argument('--n_frames', required=False, default=None, type=int)

    args = parser.parse_args()
    
    ego_path = args.ego_path
    ext_path = args.ext_path
    stream_id = args.stream_id

    print(f'ego_path: {ego_path}')
    print(f'ext_path: {ext_path}')
    print(f'stream_id: {stream_id}')
    
    video_paths = list()
    for filename in os.listdir(ext_path):
        if filename.endswith('.mp4'):
            video_path = os.path.join(ext_path, filename)
            video_paths.append(video_path)

    video_paths = natsorted(video_paths)

    caps = [cv2.VideoCapture(video_path) for video_path in video_paths]

    if len(caps) == 0:
        raise ValueError('No video files found in the ext path')

    if len(caps) <= stream_id:
        raise ValueError(f'Invalid stream id. Number of streams: {len(caps)}')

    aedat_stream_path = None
    for file_name in os.listdir(ego_path):
        if file_name.endswith('.aedat4'):
            aedat_stream_path = os.path.join(ego_path, file_name)
            break

    assert aedat_stream_path is not None, 'No aedat4 file found in the ego path'

    folder_name = ego_path.replace(os.path.sep, '_').replace(':', '_')
    folder_path = os.path.join(TEMP_DIR, folder_name)
    os.makedirs(folder_path, exist_ok=True)

    j3d_gt_path = os.path.join(folder_path, 'pose_gt.pkl')
    j3d_gt_path = parse_file(os.path.join(ext_path, 'unknown.bvh'), j3d_gt_path)


    with open(j3d_gt_path, 'rb') as f:
        pose_list = pickle.load(f)

    sync_camera = SyncCamera(ego_path, caps, aedat_stream_path, pose_list)
    sync_camera(args)


if __name__ == '__main__':
    main()
