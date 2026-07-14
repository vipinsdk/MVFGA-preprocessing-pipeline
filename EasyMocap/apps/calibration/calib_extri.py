'''
  @ Date: 2021-03-02 16:13:03
  @ Author: Qing Shuai
  @ LastEditors: Qing Shuai
  @ LastEditTime: 2022-08-03 17:35:16
  @ FilePath: /EasyMocapPublic/apps/calibration/calib_extri.py
'''
from easymocap.mytools.camera_utils import write_intri
import os
from glob import glob
from os.path import join
import numpy as np
import cv2
from tqdm import trange, tqdm
from easymocap.mytools import read_intri, write_extri, read_json
from easymocap.mytools.debug_utils import mywarn
from concurrent.futures import ThreadPoolExecutor, as_completed


def estimate_extrinsics(cam1_intrinsics, cam1_extrinsics, image_points1, image_points2):
    """
    Estimates the extrinsic parameters of Camera 2 given Camera 1's extrinsics.
    
    Args:
        cam1_intrinsics (numpy.ndarray): 3x3 intrinsic matrix of Camera 1.
        cam1_extrinsics (tuple): (R1, t1), rotation (3x3) and translation (3x1) of Camera 1.
        image_points1 (numpy.ndarray): Nx2 array of matched feature points in Camera 1.
        image_points2 (numpy.ndarray): Nx2 array of matched feature points in Camera 2.
    
    Returns:
        R2 (numpy.ndarray): 3x3 rotation matrix of Camera 2.
        t2 (numpy.ndarray): 3x1 translation vector of Camera 2.
    """
    R1, t1 = cam1_extrinsics

    # Compute the essential matrix
    E, _ = cv2.findEssentialMat(image_points1, image_points2, cam1_intrinsics, method=cv2.RANSAC, threshold=1.0)

    # Recover the relative pose between Camera 1 and Camera 2
    _, R_1to2, t_1to2, _ = cv2.recoverPose(E, image_points1, image_points2, cam1_intrinsics)

    # Compute Camera 2 extrinsics relative to world
    R2 = R_1to2 @ R1
    t2 = R_1to2 @ t1 + t_1to2

    return R2, t2

def init_intri(path, image):
    camnames = sorted(os.listdir(join(path, image)))
    cameras = {}
    for ic, cam in enumerate(camnames):
        imagenames = sorted(glob(join(path, image, cam, '*.jpg')))
        assert len(imagenames) > 0
        imgname = imagenames[0]
        img = cv2.imread(imgname)
        height, width = img.shape[0], img.shape[1]
        focal = 1.2*max(height, width) # as colmap
        K = np.array([focal, 0., width/2, 0., focal, height/2, 0. ,0., 1.]).reshape(3, 3)
        dist = np.zeros((1, 5))
        cameras[cam] = {
            'K': K,
            'dist': dist
        }
    return cameras

def solvePnP(k3d, k2d, K, dist, flag, tryextri=False):
    k2d = np.ascontiguousarray(k2d[:, :2])
    # try different initial values:
    if tryextri:
        def closure(rvec, tvec):
            ret, rvec, tvec = cv2.solvePnP(k3d, k2d, K, dist, rvec, tvec, True, flags=flag)
            points2d_repro, xxx = cv2.projectPoints(k3d, rvec, tvec, K, dist)
            kpts_repro = points2d_repro.squeeze()
            err = np.linalg.norm(points2d_repro.squeeze() - k2d, axis=1).mean()
            return err, rvec, tvec, kpts_repro
        # create a series of extrinsic parameters looking at the origin
        height_guess = 2.1
        radius_guess = 7.
        infos = []
        for theta in np.linspace(0, 2*np.pi, 180):
            st = np.sin(theta)
            ct = np.cos(theta)
            center = np.array([radius_guess*ct, radius_guess*st, height_guess]).reshape(3, 1)
            R = np.array([
                [-st, ct,  0],
                [0,    0, -1],
                [-ct, -st, 0]
            ])
            tvec = - R @ center
            rvec = cv2.Rodrigues(R)[0]
            err, rvec, tvec, kpts_repro = closure(rvec, tvec)
            infos.append({
                'err': err,
                'repro': kpts_repro,
                'rvec': rvec,
                'tvec': tvec
            })
        infos.sort(key=lambda x:x['err'])
        err, rvec, tvec, kpts_repro = infos[0]['err'], infos[0]['rvec'], infos[0]['tvec'], infos[0]['repro']
    else:
        ret, rvec, tvec = cv2.solvePnP(k3d, k2d, K, dist, flags=flag)
        points2d_repro, xxx = cv2.projectPoints(k3d, rvec, tvec, K, dist)
        kpts_repro = points2d_repro.squeeze()
        err = np.linalg.norm(points2d_repro.squeeze() - k2d, axis=1).mean()
    # print(err)
    return err, rvec, tvec, kpts_repro

def calib_extri(path, image, intriname, image_id):
    camnames = sorted(os.listdir(join(path, image)))
    camnames = [c for c in camnames if os.path.isdir(join(path, image, c))]
    if intriname is None:
        # initialize intrinsic parameters
        intri = init_intri(path, image)
    else:
        assert os.path.exists(intriname), intriname
        intri = read_intri(intriname)
        if len(intri.keys()) == 1:
            key0 = list(intri.keys())[0]
            for cam in camnames:
                intri[cam] = intri[key0].copy()
    extri = {}
    err_cam = {}
    # methods = [cv2.SOLVEPNP_ITERATIVE, cv2.SOLVEPNP_P3P, cv2.SOLVEPNP_AP3P, cv2.SOLVEPNP_EPNP, cv2.SOLVEPNP_DLS, cv2.SOLVEPNP_IPPE, cv2.SOLVEPNP_SQPNP]
    methods = [cv2.SOLVEPNP_ITERATIVE]
    for ic, cam in enumerate(camnames):
        imagenames = sorted(glob(join(path, image, cam, '*{}'.format(args.ext))))
        chessnames = sorted(glob(join(path, 'chessboard', cam, '*.json')))
        # chessname = chessnames[0]
        assert len(chessnames) > 0, cam
        chessname = chessnames[image_id]
        
        data = read_json(chessname)
        k3d = np.array(data['keypoints3d'], dtype=np.float32)
        k2d = np.array(data['keypoints2d'], dtype=np.float32)
        if k3d.shape[0] != k2d.shape[0]:
            mywarn('k3d {} doesnot match k2d {}'.format(k3d.shape, k2d.shape))
            length = min(k3d.shape[0], k2d.shape[0])
            k3d = k3d[:length]
            k2d = k2d[:length]
        #k3d[:, 0] *= -1
        valididx = k2d[:, 2] > 0
        if valididx.sum() < 4:
            extri[cam] = {}
            rvec = np.zeros((1, 3))
            tvec = np.zeros((3, 1))
            extri[cam]['Rvec'] = rvec
            extri[cam]['R'] = cv2.Rodrigues(rvec)[0]
            extri[cam]['T'] = tvec
            print('[ERROR] Failed to initialize the extrinsic parameters')
            extri.pop(cam)
            continue
        k3d = k3d[valididx]
        k2d = k2d[valididx]
        if args.tryfocal:
            infos = []
            for focal in range(500, 5000, 10):
                dist = intri[cam]['dist']
                K = intri[cam]['K']
                K[0, 0] = focal
                K[1, 1] = focal
                for flag in methods:
                    err, rvec, tvec, kpts_repro = solvePnP(k3d, k2d, K, dist, flag)
                    infos.append({
                        'focal': focal,
                        'err': err,
                        'repro': kpts_repro,
                        'rvec': rvec,
                        'tvec': tvec
                    })
            infos.sort(key=lambda x:x['err'])
            err, rvec, tvec = infos[0]['err'], infos[0]['rvec'], infos[0]['tvec']
            kpts_repro = infos[0]['repro']
            focal = infos[0]['focal']
            intri[cam]['K'][0, 0] = focal
            intri[cam]['K'][1, 1] = focal
        else:
            K, dist = intri[cam]['K'], intri[cam]['dist']
            err, rvec, tvec, kpts_repro = solvePnP(k3d, k2d, K, dist, flag=cv2.SOLVEPNP_ITERATIVE)
        extri[cam] = {}
        extri[cam]['Rvec'] = rvec
        extri[cam]['R'] = cv2.Rodrigues(rvec)[0]
        extri[cam]['T'] = tvec
        center = - extri[cam]['R'].T @ tvec
        err_cam[cam] = err
        print('{} center => {}, err = {:.3f}'.format(cam, center.squeeze(), err))

    return {
        'intri': intri,
        'extri': extri,
        'err_cam': err_cam,
        'global_err': max(list(err_cam.values())) if len(err_cam) > 0 else 1e6,
        'image_id': image_id
    }

def process_image(image_id, path, image, intriname, camnames):
    result = calib_extri(path, image, intriname=intriname, image_id=image_id)
    all_cams_exist = all(cam in result['extri'] for cam in camnames)
    return result if all_cams_exist else None


def calibrate_extrinsics(path, image, intriname):    
    camnames = sorted(os.listdir(join(path, image)))
    camnames = [c for c in camnames if os.path.isdir(join(path, image, c))]
    n_images = min([len(glob(join(path, image, cam, '*.jpg'))) for cam in camnames])

    calib_params = []

    with ThreadPoolExecutor() as executor:
        futures = {
            executor.submit(process_image, i, path, image, intriname, camnames): i
            for i in range(n_images)
        }

        for future in tqdm(as_completed(futures), total=n_images):
            result = future.result()
            if result is not None:
                calib_params.append(result)

    if not calib_params:
        print("No valid calibration parameters found.")
        return

    min_error = np.argsort([x['global_err'] for x in calib_params])
    best_params = calib_params[min_error[0]]
    
    image_id = best_params['image_id']
    intri = best_params['intri']
    extri = best_params['extri']

    write_intri(join(path, 'intri.yml'), intri)
    write_extri(join(path, 'extri.yml'), extri, image_id)
    
    print('Best image:', image_id)
    print('Best error:', best_params['global_err'])
    print('Average error:', np.mean([x['global_err'] for x in calib_params]))
    print('Worst error:', np.max([x['global_err'] for x in calib_params]))


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('path', type=str)
    parser.add_argument('--image', type=str, default='images')
    parser.add_argument('--intri', type=str, default=None)
    parser.add_argument('--ext', type=str, default='.jpg')
    parser.add_argument('--step', type=int, default=1)
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--tryfocal', action='store_true')
    parser.add_argument('--tryextri', action='store_true')
    parser.add_argument('--image_id', type=int, default=0, help='Image id used for extrinsic calibration')

    args = parser.parse_args()
    calibrate_extrinsics(args.path, args.image, intriname=args.intri)