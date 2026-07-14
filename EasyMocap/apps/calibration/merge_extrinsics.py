import cv2
import numpy as np
from os.path import join
from easymocap.mytools.camera_utils import write_intri
from easymocap.mytools import read_intri, write_extri, read_extri

def convert_world_2_to_world_1(extri1, extri2_path, camnames):
    extri2 = read_extri(join(extri2_path, 'extri.yml'))
    w1_2_c1 = np.eye(4)
    w2_2_c2 = np.eye(4)
    w2_2_c1 = np.eye(4)

    for cam in camnames:
        if cam in extri1 and cam in extri2:
            R1_w1 = extri1[cam]['R']
            T1_w1 = extri1[cam]['T']

            R1_w2 = extri2[cam]['R']
            T1_w2 = extri2[cam]['T']

            w1_2_c1 = np.vstack((np.hstack((R1_w1, T1_w1)), [0, 0, 0, 1]))
            w2_2_c1 = np.vstack((np.hstack((R1_w2, T1_w2)), [0, 0, 0, 1]))

        if cam in extri2:
            R2_w2 = extri2[cam]['R']
            T2_w2 = extri2[cam]['T']

            w2_2_c2 = np.vstack((np.hstack((R2_w2, T2_w2)), [0, 0, 0, 1]))

    c1_2_w1 = np.linalg.inv(w1_2_c1)
    c2_2_w2 = np.linalg.inv(w2_2_c2)

    c2_2_w1 = c1_2_w1 @ w2_2_c1 @ c2_2_w2
    w1_2_c2 = np.linalg.inv(c2_2_w1)

    cam = {}
    cam['R'] = w1_2_c2[:3, :3]
    cam['Rvec'] = cv2.Rodrigues(w1_2_c2[:3, :3])[0]
    cam['T'] = np.array([w1_2_c2[:3, 3]])
    return cam

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('path', type=str)
    parser.add_argument('extri1_path', type=str)
    parser.add_argument('extri2_path', type=str)
    
    args = parser.parse_args()
    path = join(args.path, 'extrinsics')
    extri1_path = join(args.extri1_path, 'extrinsics')
    extri2_path = join(args.extri2_path, 'extrinsics')

    intrinsics = read_intri(join(path, 'intri.yml'))
    extrinsics = read_extri(join(path, 'extri.yml'))

    intri_1 = read_intri(join(extri1_path, 'intri.yml'))
    intri_2 = read_intri(join(extri2_path, 'intri.yml'))

    for key, val in intri_2.items():
        if key not in intrinsics:
            intrinsics[key] = {}
            intrinsics[key]['K'] = val['K']
            intrinsics[key]['dist'] = val['dist']

    for key, val in intri_1.items():
        if key not in intrinsics:
            intrinsics[key] = {}
            intrinsics[key]['K'] = val['K']
            intrinsics[key]['dist'] = val['dist']

    write_intri(join(args.path, 'videos', 'intri.yml'), intrinsics)

    camnames = [['013', '016'], ['001', '017']]
    extri_paths = [extri2_path, extri1_path]
    for extri, view in zip(extri_paths,camnames):
        cam = convert_world_2_to_world_1(extrinsics, extri, view)
        extrinsics[view[1]] = cam
    
    write_extri(join(args.path, 'videos', 'extri.yml'), extrinsics)

    # move the cameras.json file to root directory
    # cameras_json_out = join(path, 'camera_params.json')
    # os.rename(cameras_json_out, join(args.path, 'camera_params.json'))