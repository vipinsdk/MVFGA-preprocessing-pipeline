#!/bin/bash

data=$1
pose_out=$2
out=$3

mkdir -p ${pose_out}
mkdir -p ${out}

echo "install dependencies"
apt-get install libosmesa6-dev -y
pip uninstall pyopengl -y && pip install pyopengl==3.1.5


echo "extracting video"
python3 scripts/preprocess/extract_video.py ${data} --no2d


echo "Run mediapipe"
python3 apps/preprocess/extract_keypoints.py ${data} --img_save ${pose_out} --mode mp-holistic

echo "Fit SMPLX"
PYOPENGL_PLATFORM=osmesa python3 apps/demo/mv1p.py ${data} --out ${out} --vis_det --vis_repro --undis --body bodyhandface --model smplx --gender male --vis_smpl

# PYOPENGL_PLATFORM=osmesa python3 apps/demo/mv1p.py /netscratch/jeetmal/videos/six_cam/ --out /netscratch/jeetmal/output/easymocap/mediapipe/handr/six_cam --vis_det --vis_repro --undis --body handr --model manor --vis_smpl
# PYOPENGL_PLATFORM=osmesa python3 apps/demo/mv1p.py /netscratch/jeetmal/videos/Ameer/ --out /netscratch/jeetmal/output/easymocap/mediapipe/handl/Ameer/handface --vis_det --vis_repro --undis --body handl --model manol --gender male --vis_smpl --sub 1 2 3 4 5 6 --sub_vis 2 --mesh_root /netscratch/j




















