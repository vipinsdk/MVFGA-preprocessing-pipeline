import os
from npybvh.bvh import Bvh
import numpy as np
from tqdm import trange
import pickle


# from utils.skeleton import Skeleton
# skeleton_model = Skeleton(None)
# egocentric_joints = [6, 15, 16, 17, 10, 11, 12, 23, 24, 25, 26, 19, 20, 21, 22]


def parse_file(bvh_file_path, output_file_path):
    anim = Bvh()

    anim.parse_file(bvh_file_path)
    gt_pose_seq = []
    print(anim.frames)
    print(anim.joint_names())
    joint_name_list = list(anim.joint_names())
    egocentric_joints = [joint_name_list.index('Head'),
                         joint_name_list.index('Neck'),
                         joint_name_list.index('RightArm'),
                         joint_name_list.index('RightForeArm'),
                         joint_name_list.index('RightHand'),
                         joint_name_list.index('LeftArm'),
                         joint_name_list.index('LeftForeArm'),
                         joint_name_list.index('LeftHand'),
                         joint_name_list.index('RightUpLeg'),
                         joint_name_list.index('RightLeg'),
                         joint_name_list.index('RightFoot'),
                         joint_name_list.index('RightToeBase'),
                         joint_name_list.index('LeftUpLeg'),
                         joint_name_list.index('LeftLeg'),
                         joint_name_list.index('LeftFoot'),
                         joint_name_list.index('LeftToeBase'), ]
    
    if os.path.exists(output_file_path):
        print(f'Loading existing file from path: {output_file_path}')
        with open(output_file_path, 'rb') as f:
            gt_pose_seq = pickle.load(f)
        return output_file_path
    
    for frame in trange(len(anim.keyframes)):
        positions, rotations = anim.frame_pose(frame)

        positions = positions[egocentric_joints]
        positions = positions / 1000
        gt_pose_seq.append(positions)

        # skeleton = skeleton_model.joints_2_mesh(positions)
        #
        # open3d.visualization.draw_geometries([skeleton])
        
    gt_pose_seq = np.asarray(gt_pose_seq)
    # skeleton_list = skeleton_model.joint_list_2_mesh_list(gt_pose_seq)
    # open3d.visualization.draw_geometries(skeleton_list)
    
    with open(output_file_path, 'wb') as f:
        pickle.dump(gt_pose_seq, f)

    return output_file_path


if __name__ == '__main__':
    # parse_file(r'\\winfs-inf\HPS\Mo2Cap2Plus1\static00\CapturyData\GoProResults\captury\unknown.bvh', 'data/wild/wild.pkl', start_frame=0, input_frame_rate=50, output_frame_rate=25)
    parse_file(r'\\winfs-inf\CT\EgoEvents\nobackup\Recording_09_05_23\christen_09_05_23_captury\unknown.bvh',
               r'pose_gt.pkl', start_frame=0)
