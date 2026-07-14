import os
import argparse
import logging
import sys
from preprocess.trim_synced_streams import trim_videos
from BiRefNet.demo import background_matting
from preprocess.extract_video import extract_video

# Setup logging configuration
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),  # Prints to console
        logging.FileHandler('output.log')  # Logs to a file named 'output.log'
    ]
)

def run_command(command):
    """Function to run a shell command and check for errors."""
    logging.info(f"Running command: {command}")
    # Use os.system to execute the command and check the return code
    ret_code = os.system(command)
    if ret_code != 0:
        logging.error(f"Error occurred while executing: {command}")
        sys.exit(ret_code)  # Exit the script if command fails

def calibrate_cameras(args, root_dir):
    # Variables (these would normally come from user input or environment variables)
    base_path = root_dir
    pattern = args.pattern
    grid = args.grid
    grid_step = args.grid_step
    intri = args.list[0]
    extri = args.list[1]

    #  check if the calibration files already exist
    if os.path.exists(os.path.join(base_path, extri, 'extri.yml')) and os.path.exists(os.path.join(base_path, extri, 'intri.yml')):
        logging.info(f"Calibration files already exist in {base_path}/{extri}.")
        return
    
    # change the directory to the Easymocap code
    os.chdir('EasyMocap')
    for item in args.list:
        # Command 0: extract individual frames from the video
        logging.info(f"Extracting frames from {item}")
        extract_frames_command = f" python3 scripts/preprocess/extract_video.py {base_path}/{item} --no2d"
        run_command(extract_frames_command)

        # Command 1: Detect the chessboard in all cameras
        logging.info(f"Detecting chessboard in {item}")
        detect_chessboard_command = f"python3 apps/calibration/detect_chessboard.py {base_path}/{item} --out {base_path}/{item}/output/calibration --pattern {pattern} --grid {grid}"
        run_command(detect_chessboard_command)


    # Command 2: Intrinsic camera calibration
    calib_intri_command = f"python3 apps/calibration/calib_intri.py {base_path}/{intri}"
    run_command(calib_intri_command)

    # Command 3: Extrinsic camera calibration
    calib_extri_command = f"python3 apps/calibration/calib_extri.py {base_path}/{extri} --intri {base_path}/{intri}/output/intri.yml"
    run_command(calib_extri_command)

    # Command 4: Check calibration
    check_calib_command = f"python3 apps/calibration/check_calib.py {base_path}/{extri} --out {base_path}/{extri} --mode cube --write --grid_step {grid_step}"
    run_command(check_calib_command)

    # change to the root directory
    os.chdir('..')

def background_mask(input_path, output_path, seg_path):
    if os.path.exists(output_path) and os.listdir(os.path.join(output_path, 'images')):
        logging.info(f"Output folder {output_path} already exists.")
    else:
        os.makedirs(output_path, exist_ok=True)
    background_matting(input_path, output_path, seg_path)
    

def body_segmentation(input_path, output_path, model_path):
    os.chdir('sapiens/lite/scripts/demo/torchscript')
    if os.path.exists(output_path) and os.listdir(output_path):
        logging.info(f"Output folder {output_path} already exists.")
        os.chdir('../../../../../..')
    else:
        os.makedirs(output_path, exist_ok=True)
        # Command 0: Run Sapiens
        sapiens_command = f"bash seg.sh {input_path} {output_path} {model_path}"
        run_command(sapiens_command)
        os.chdir('../../../../../..')

def main(args):
    base_path = args.root_dir

    if args.trim:
        # Loop through the items in the list
        list.append('videos')
        for item in args.list:
            if item not in os.listdir(base_path):
                logging.error(f"Item {item} not found in the root directory.")
                sys.exit(1)

            input = os.path.join(base_path, item)
            # Command 0: Trim videos
            trim_videos(input, args.output)
    
    if args.calibrate:
        calibrate_cameras(args, base_path)

        ext_1 = os.path.join(base_path, 'extrinsics_1')
        ext_2 = os.path.join(base_path, 'extrinsics_2')

        if os.path.exists(ext_1):
            calibrate_cameras(args, ext_1)
        if os.path.exists(ext_2):
            calibrate_cameras(args, ext_2)

        # merge extrinsics
        if os.path.exists(ext_1) and os.path.exists(ext_2):
            os.chdir('EasyMocap')
            logging.info("Merging extrinsics")
            merge_command = f"python3 apps/calibration/merge_extrinsics.py {base_path} {ext_1} {ext_2}"
            run_command(merge_command)
            os.chdir('..')
    
    input = os.path.join(base_path, 'videos')
    assert os.path.exists(input), f"Input directory {input} does not exist."

    source = os.path.join(input, 'source')
    if not os.path.exists(source):
        extract_video(input, target_fps=25)
    else:
        logging.info("Frames already extracted.")

    if args.annots:
        logging.info("Generating annotations using Mediapipe")
        os.chdir('EasyMocap')
        videos = os.path.join(base_path, 'videos')
        annots_command = f"python3 apps/preprocess/extract_keypoints.py {videos} --mode {args.mode} --output {args.output} --pose_save {args.pose_save}"
        run_command(annots_command)
        os.chdir('..')

    if args.sapiens:
        assert os.path.exists(source), f"Input directory {source} does not exist."
        output_folder = os.path.join(args.output, 'BodySegmentation')
        model_path = args.sapiens_lite_ckpt
        body_segmentation(source, output_folder, model_path)

    if args.background_matting:
        model_path = 'BiRefNet/models/BiRefNet-general-epoch_244.pth'
        if not os.path.exists(model_path):
            os.system('wget https://github.com/ZhengPeng7/BiRefNet/releases/download/v1/BiRefNet-general-epoch_244.pth -o BiRefNet/models/BiRefNet-general-epoch_244.pth')

        assert os.path.exists(source), f"Input directory {source} does not exist."
        output_folder = os.path.join(args.output, 'background_matting')
        seg_folder = os.path.join(args.output, 'BodySegmentation', 'sapiens_1b')
        background_mask(source, output_folder, seg_folder)
    
    if args.flame_params:
        logging.info("Generating FLAME parameters")
        if args.flame_config is None:
            logging.error("Please provide a config file for metrical tracker.")
            sys.exit(1)
        os.chdir('metrical-tracker_multiview')
        flame_command = f"python tracker.py --cfg configs/actors/{args.flame_config}"
        run_command(flame_command)
    
    if args.mano_params:
        logging.info("Generating MANO parameters")
        os.chdir('hamer')
        hamer_out = os.path.join(args.output, 'hamer')
        source = "source"
        root_folder = os.path.join(args.root_dir, 'videos')
        mano_command = f"python demo_multi.py --root_folder {root_folder} --img_folder {source} --out_folder {hamer_out} --batch_size=48 --save_mesh --full_frame --save_params"
        run_command(mano_command)

    if args.smplx_params:
        logging.info("Generate SMPLX parameters")
        os.chdir('EasyMocap')
        smplx_out = os.path.join(args.output, 'easymocap')
        if os.path.exists(smplx_out):
            logging.info(f"Output folder {smplx_out} already exists. Making a new folder")
            smplx_out = os.path.join(args.output, 'easymocap_1')
        root_folder = os.path.join(args.root_dir, 'videos')
        smplx_command = f"python3 apps/demo/mv1p.py {root_folder} --out {smplx_out} --vis_det --vis_repro --body {args.body} --model {args.smplx} --gender {args.gender} --vis_smpl --verbose"

        if args.flame_out is not None:
            smplx_command += f" --flame_path {args.flame_out}"
        if args.mano_out is not None:
            smplx_command += f" --mano_path {args.mano_out}"
        run_command(smplx_command)
        os.chdir('..')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Data preprocessing for MVFGA dataset')
    parser.add_argument("--root_dir", type=str, required=True, help="Path to the root directory")
    parser.add_argument("--output", type=str, help="Path to the output folder")
    parser.add_argument('--list', nargs='+', default=['intrinsics', 'extrinsics'], help="List of items enclosed in brackets (required)")
    parser.add_argument('--pattern', type=str, default='8,13', help="Pattern in the format '8,13'")
    parser.add_argument('--grid', type=float, default=0.04, help="Grid dimension for the checkerboard")
    parser.add_argument('--grid_step', type=float, default=0.24 ,help="Grid step value for the checkerboard (cube size)")
    parser.add_argument('--trim', action='store_true', help="Trim videos")
    parser.add_argument('--calibrate', action='store_true', help="Calibrate cameras")
    parser.add_argument('--background_matting', action='store_true', help="Run background matting")
    parser.add_argument('--sapiens', action='store_true', help="Run Sapiens")
    parser.add_argument('--sapiens_lite_ckpt', type=str, default='', help="Path to the Sapiens Lite checkpoint")
    parser.add_argument('--annots', action='store_true', help="To generate annotations (mediapipe)")
    parser.add_argument('--pose_save', type=bool, default=False, help="To save pose output")
    parser.add_argument('--mode', type=str, default='mp-holistic', help="Mode for generating annotations")
    parser.add_argument('--flame_params', action='store_true', help="To generate flame parameters")
    parser.add_argument('--flame_config', type=str, default=None, help="config file for metrical tracker")
    parser.add_argument('--mano_params', action='store_true', help="To generate mano parameters")
    parser.add_argument('--smplx_params', action='store_true', help="To generate smplx parameters")
    parser.add_argument('--flame_out', type=str, help="To use FLAME parameters")
    parser.add_argument('--mano_out', type=str, help="To use MANO parameters")
    parser.add_argument('--body', type=str, default='bodyhandface', help="Body parts to use")
    parser.add_argument('--smplx', type=str, default='smplx', help="SMPLX model to use")
    parser.add_argument('--gender', type=str, default='male', help='gender')
    args = parser.parse_args()
    main(args)
