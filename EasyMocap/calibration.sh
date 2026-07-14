#!/bin/bash

# Function to show usage information
usage() {
  echo "Usage: $0 --list '[intri, extri]' [--pattern 8,13] [--grid 0.04] [--grid_step 0.24]"
  echo "  --list: List of items enclosed in brackets (required)"
  exit 1
}

# Initialize variables with default values
base_path=/netscratch/jeetmal/videos/Ameer_full_setup
calib_list=""
pattern=""
grid=""
intri=""
extri=""
grid_step=""

echo $1
# Parse command-line options
while [[ $# -gt 0 ]]; do
  case $1 in
    --list)
      calib_list="$2"
      shift 2
      ;;
    --pattern)
      pattern="$2"
      shift 2
      ;;
    --grid)
      grid="$2"
      shift 2
      ;;
    --grid_step)
      grid_step="$2"
      shift 2
      ;;
    *)
      usage
      ;;
  esac
done

# Check if the input list is provided
if [ -z "$calib_list" ]; then
  usage
fi

# Remove the square brackets
calib_list="${calib_list#[}"
calib_list="${calib_list%]}"

# Split the list into an array
IFS=',' read -ra items <<< "$calib_list"

# Loop over each item
for item in "${items[@]}"
do
  # Trim leading and trailing whitespace
  item=$(echo "$item" | xargs)
  # Perform some operation on each item
  # Here we simply print the item
  echo "Processing item: $item"
  if [[ "$item" == *"intrinsic"* ]]; then
    intri=$item
  else
    extri=$item
  fi

  echo "Camera calibration pipeline from the Easymoap"
  echo "Get the frames from individual frames"
  python3 scripts/preprocess/extract_video.py $base_path/$item --no2d

  echo "Detect the chessboard in all cameras"
  python3 apps/calibration/detect_chessboard.py $base_path/$item --out $base_path/$item/output/calibration --pattern $pattern --grid $grid
done

echo "Intrinsic camera calibration"
python3 apps/calibration/calib_intri.py $base_path/$intri

echo "Extrinsic camera calibration"
python3 apps/calibration/calib_extri.py $base_path/$extri --intri $base_path/$intri/output/intri.yml

python3 apps/calibration/check_calib.py $base_path/$extri --out $base_path/$extri --mode cube --write --grid_step $grid_step

echo "Done!!!"

# python3 apps/preprocess/extract_keypoints.py ${data} --mode mp-holistic

# a=0; for i in *.jpg; do mv "$i" $(printf "%06d.jpg" $a); a=$((a+1)); done