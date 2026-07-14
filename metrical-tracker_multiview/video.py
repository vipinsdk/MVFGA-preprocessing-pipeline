import os
from pathlib import Path
import tyro
import ffmpeg

def video2frames(video_path: Path, image_dir: Path, keep_video_name: bool=False, target_fps: int=30, n_downsample: int=1):
    print(f'Converting video {video_path} to frames with downsample scale {n_downsample}')
    if not image_dir.exists():
        image_dir.mkdir(parents=True)
    
    file_path_stem = video_path.stem + '_' if keep_video_name else ''

    probe = ffmpeg.probe(str(video_path))
    
    video_fps = int(probe['streams'][0]['r_frame_rate'].split('/')[0])
    if  video_fps ==0:
        video_fps = int(probe['streams'][0]['avg_frame_rate'].split('/')[0])
        if video_fps == 0:
            # nb_frames / duration
            video_fps = int(probe['streams'][0]['nb_frames']) / float(probe['streams'][0]['duration'])
            if video_fps == 0:
                raise ValueError('Cannot get valid video fps')

    num_frames = int(probe['streams'][0]['nb_frames'])
    video = next((stream for stream in probe['streams'] if stream['codec_type'] == 'video'), None)
    W = int(video['width'])
    H = int(video['height'])
    w = W // n_downsample
    h = H // n_downsample
    print(f'[Video]  FPS: {video_fps} | number of frames: {num_frames} | resolution: {W}x{H}')
    print(f'[Target] FPS: {target_fps} | number of frames: {round(num_frames * target_fps / int(video_fps))} | resolution: {w}x{h}')

    (ffmpeg
    .input(str(video_path))
    .filter('fps', fps=f'{target_fps}')
    .filter('scale', width=w, height=h)
    .output(
        str(image_dir / f'{file_path_stem}%06d.jpg'),
        start_number=0,
        qscale=1,  # lower values mean higher quality (1 is the best, 31 is the worst).
    )
    .overwrite_output()
    .run(quiet=True)
    )
    
def main(
        input: Path, 
        target_fps: int=30, 
    ):

    if not input.exists():
        matched_paths = list(input.parent.glob(f"{input.name}*"))
        if len(matched_paths) == 0:
            raise FileNotFoundError(f"Cannot find the directory: {input}")
        elif len(matched_paths) == 1:
            input = matched_paths[0]
        else:
            raise FileNotFoundError(f"Found multiple matched folders: {matched_paths}")
            
    # prepare path
    if input.suffix in ['.mov', '.mp4']:
        print(f'Processing video file: {input}')
        videos = [input]
        image_dir = input.parent / input.stem / 'source'
    elif input.is_dir():
        # if input is a directory, assume all contained videos are synchronized multiview of the same scene
        print(f'Processing directory: {input}')
        videos = list(input.glob('*.mp4'))
        image_dir = input / 'source'
    else:
        raise ValueError(f"Input should be a video file or a directory containing video files: {input}")

    # extract frames
    for i, video_path in enumerate(videos):
        print(f'\n[{i+1}/{len(videos)}] Processing video file: {video_path}')
        video2frames(video_path, image_dir, keep_video_name=len(videos) > 1, target_fps=target_fps)
        
if __name__ == '__main__':
    tyro.cli(main)

    # Example usage:
    # python video.py /path/to/video.mp4 --target_fps 30