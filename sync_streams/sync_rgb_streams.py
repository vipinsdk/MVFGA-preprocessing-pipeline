import argparse
import os
import cv2
import imutils
import numpy as np

from config import TEMP_DIR
from tqdm import tqdm
from natsort import natsorted

from utils.utils import process_video_camera, load_camera_model, video_to_image, get_multi_view_frame
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor


RGB_START = 411
EVENT_START = 128


class SyncCamera:        
    def __init__(self, rgb_stream, caps):
        self.rgb_stream = rgb_stream
        self.caps = caps  
                
    def get_fine_rgb_sync_frame_id(self):
        print('''W -> Forward RGB Sequence, S -> Backward RGB Sequence, D -> Forward Event Sequence, A -> Backward Event Sequence, 
                'G -> Forward 10 RGB Sequence, F -> Backward 10 RGB Sequence, B -> Forward 10 Event Sequence, V -> Backward 10 Event Sequence
                q - Set sync frame start.
                p - Forward 100 RGB Sequence, 100 Event Sequence 
                [,] to change stream
                ''')

        rgb_stream = cv2.VideoCapture(self.rgb_stream)

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
            cap.set(cv2.CAP_PROP_POS_FRAMES, findex)

            if not ret: break        
            frame = imutils.resize(frame, width=1080)
            
            rgb_stream.set(cv2.CAP_PROP_POS_FRAMES, eindex)
            ret, eframe = rgb_stream.read()
            rgb_stream.set(cv2.CAP_PROP_POS_FRAMES, eindex)

            
            cv2.imshow('eframe', eframe)
            cv2.imshow('frame', frame)
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
            
            print(f'current_frame: {findex} eindex: {eindex} stream_id: {stream_id}')
        
        self.close_streams()

        return findex, eindex, stream_id

    def get_number_of_frames(self, findex, eindex, stream_id):
        print('''W -> Forward RGB Sequence, S -> Backward RGB Sequence, 
                 F -> Fast/Slow seek mode, 
                 q -> Set number of frames.
                ''')

        cap = self.caps[stream_id]
        rgb_stream = cv2.VideoCapture(self.rgb_stream)
        
        seek_mode = 'slow'
        cap.set(cv2.CAP_PROP_POS_FRAMES, findex)
        num_frames = 0
        while True:
            cap.set(cv2.CAP_PROP_POS_FRAMES, findex + num_frames)        
            ret, frame = cap.read()
            cap.set(cv2.CAP_PROP_POS_FRAMES, findex + num_frames)
            
            if not ret: break        
            frame = imutils.resize(frame, width=1080)


            rgb_stream.set(cv2.CAP_PROP_POS_FRAMES, eindex + num_frames)
            ret, eframe = rgb_stream.read()
            rgb_stream.set(cv2.CAP_PROP_POS_FRAMES, eindex + num_frames)

            if not ret: break        

            # eframe = cv2.cvtColor(cv2.resize(eframe, (frame.shape[1], frame.shape[0])), cv2.COLOR_GRAY2BGR)

            # cv2.imshow('stacked_frame', np.hstack((frame, eframe)))

            cv2.imshow('eframe', eframe)
            cv2.imshow('frame', frame)

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
        
            print(f'Timestep: fstream: {findex + num_frames} estream: {eindex + num_frames} n_frames: {num_frames}')
            
            if ord('f') == c:                
                if seek_mode == 'fast':
                    seek_mode = 'slow'
                else: seek_mode = 'fast'
            
            if ord('q') == c:
                break
    
        return eindex

    def __call__(self, stream_id):
        if RGB_START is not None and EVENT_START is not None:
            findex, eindex = RGB_START, EVENT_START

        # else:

        # findex, eindex = 285, 116
        # n_frames = 1523

        # findex, eindex, stream_id = self.get_fine_rgb_sync_frame_id()
        # n_frames = self.get_number_of_frames(findex, eindex, stream_id)

        print(f'Sync frame: {findex} Sync eindex: {eindex} n_frames: {n_frames} stream_id: {stream_id}')

        return findex, eindex, n_frames


def main():

    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--rgb_stream', required=True, type=str)    
    parser.add_argument('--ext_path', required=True, type=str)
    parser.add_argument('--stream_id', required=True, type=int)

    args = parser.parse_args()
    
    rgb_stream = args.rgb_stream
    ext_path = args.ext_path

    video_paths = list()
    for filename in os.listdir(ext_path):
        if filename.endswith('.mp4'):
            video_path = os.path.join(ext_path, filename)
            video_paths.append(video_path)

    video_paths = natsorted(video_paths)

    caps = [cv2.VideoCapture(video_path) for video_path in video_paths]

    sync_camera = SyncCamera(rgb_stream, caps)
    findex, eindex, n_frames = sync_camera(args.stream_id)

    rgb_stream_images_path = video_to_image(rgb_stream)

    camera_calib_list = load_camera_model(ext_path)
    camera_list = []
    with ThreadPoolExecutor(8) as executor:
        for future in tqdm([executor.submit(process_video_camera, camera_item, ext_path) for camera_item in camera_calib_list]):
            camera_list.append(future.result())

    print('camera_list', camera_list)
    print('rgb_stream_images', rgb_stream_images_path)

    pbar = tqdm(total=n_frames)
    seek_mode = 'slow'
    index = 0
    update_itr = 1
    while True:
        if index >= n_frames: break
        pbar.update(update_itr)
        index += update_itr
        update_itr = 1
    
        rgb_frame = cv2.imread(os.path.join(rgb_stream_images_path, '%06d.png' % (eindex + index)))
        
        stream_frames = []
        with ThreadPoolExecutor(max_workers=len(camera_list)) as executor:
            for frame  in executor.map(lambda x:  cv2.imread(x), get_multi_view_frame(camera_list, findex + index)):
                stream_frames.append(frame)
        
        cv2.imshow('rgb_frame', rgb_frame)
        cv2.imshow('stream_frame', imutils.resize(stream_frames[0], width=1080))
        if seek_mode == 'fast':
            c = cv2.waitKey(1)
        else:
            c = cv2.waitKey(0)

        if c == ord('s'):
            print('Saving frames')
            save_frame_folder = f'output/synced_frames_{index}'
            os.makedirs(save_frame_folder, exist_ok=True)

            cv2.imwrite(f'{save_frame_folder}/rgb_frame_{index}.png', rgb_frame)
            [cv2.imwrite(f'{save_frame_folder}/stream_frame_{index}_{idx}.png', stream_frame) for idx, stream_frame in enumerate(stream_frames)]
            print('Saving frames Done')

        if c == ord('q'):
            break

        if ord('w') == c:
            update_itr = 100

        if ord('f') == c:                
            if seek_mode == 'fast':
                seek_mode = 'slow'
            else: seek_mode = 'fast'


if __name__ == '__main__':
    main()
