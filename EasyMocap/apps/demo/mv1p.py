'''
  @ Date: 2021-04-13 19:46:51
  @ Author: Qing Shuai
  @ LastEditors: Qing Shuai
  @ LastEditTime: 2021-06-13 17:56:25
  @ FilePath: /EasyMocap/apps/demo/mv1p.py
'''
from tqdm import tqdm
from easymocap.smplmodel import check_keypoints, load_model, select_nf
from easymocap.mytools import simple_recon_person, Timer, projectN3
from easymocap.pipeline import smpl_from_keypoints3d2d

import os
import trimesh
from os.path import join
import numpy as np
import torch
import json
import torchvision
import cv2
from PIL import Image
import torchvision.transforms.functional as F_t
import torchvision.transforms as T
from pytorch3d.transforms import rotation_6d_to_matrix, matrix_to_euler_angles

def find_boundary_edges(faces):
    """Find edges that appear only once (boundary edges)."""
    edges = torch.cat([
        faces[:, [0, 1]],
        faces[:, [1, 2]],
        faces[:, [2, 0]]
    ])
    edges = torch.sort(edges, dim=1)[0]
    unique_edges, counts = torch.unique(edges, dim=0, return_counts=True)
    return unique_edges[counts == 1]

def remove_vertices(vertices,faces,vertices_to_remove,fill_holes: bool = True) -> tuple:
    vertices = torch.from_numpy(vertices).cuda()
    faces = torch.from_numpy(np.array(faces, dtype=np.int32)).long().cuda()

    """
    Remove vertices and optionally fill resulting holes in the mesh.
    
    Args:
        vertices: Tensor(N, 3) - Vertex coordinates
        faces: Tensor(M, 3) - Face indices
        vertices_to_remove: Tensor(K) - Indices to remove
        fill_holes: bool - Whether to fill holes after removal
        
    Returns:
        new_vertices: Updated vertices
        new_faces: Updated faces including hole fills
    """
    device = vertices.device
    
    # Remove vertices (same as before)
    keep_mask = torch.ones(vertices.size(0), dtype=torch.bool, device=device)
    keep_mask[vertices_to_remove] = False
    vertex_map = torch.cumsum(keep_mask, dim=0) - 1
    vertex_map[~keep_mask] = -1
    new_vertices = vertices[keep_mask]
    valid_faces_mask = torch.all(vertex_map[faces] != -1, dim=1)
    new_faces = vertex_map[faces[valid_faces_mask]]
    
    if not fill_holes:
        return new_vertices, new_faces
    
    # Find boundary edges
    boundary_edges = find_boundary_edges(new_faces)
    
    if len(boundary_edges) == 0:
        return new_vertices.unsqueeze(0), new_faces
        
    # Create new vertices and faces to fill holes
    additional_vertices = []
    additional_faces = []
    
    # Process each connected boundary loop
    processed_edges = set()
    
    while len(processed_edges) < len(boundary_edges):
        # Find a starting edge
        for edge in boundary_edges:
            edge_tuple = tuple(edge.tolist())
            if edge_tuple not in processed_edges:
                start_edge = edge
                break
                
        # Trace boundary loop
        current_loop = [start_edge[0].item()]
        current_edge = start_edge
        
        while True:
            processed_edges.add(tuple(current_edge.tolist()))
            next_vertex = current_edge[1].item()
            current_loop.append(next_vertex)
            
            # Find next edge in loop
            next_edges = boundary_edges[boundary_edges[:, 0] == next_vertex]
            if len(next_edges) == 0:
                next_edges = boundary_edges[boundary_edges[:, 1] == next_vertex]
                if len(next_edges) == 0:
                    break
                    
            found_next = False
            for next_edge in next_edges:
                if tuple(next_edge.tolist()) not in processed_edges:
                    current_edge = next_edge
                    found_next = True
                    break
                    
            if not found_next:
                break
        
        if len(current_loop) >= 3:
            # Create center vertex (average of boundary vertices)
            center = torch.mean(new_vertices[current_loop], dim=0)
            center_idx = len(new_vertices) + len(additional_vertices)
            additional_vertices.append(center)
            
            # Create fan triangulation
            for i in range(len(current_loop) - 1):
                additional_faces.append([current_loop[i], current_loop[i + 1], center_idx])
            additional_faces.append([current_loop[-1], current_loop[0], center_idx])
    
    # Combine original and new geometry
    if additional_vertices:
        additional_vertices = torch.stack(additional_vertices)
        final_vertices = torch.cat([new_vertices, additional_vertices])
        additional_faces = torch.tensor(additional_faces, device=device)
        final_faces = torch.cat([new_faces, additional_faces])
        return final_vertices.unsqueeze(0), final_faces
    
    return new_vertices.unsqueeze(0), new_faces

def find_vertices_to_remove(vertices_ndc, binary_mask):
    vertices_ndc = vertices_ndc[..., :2]
    seg_path = os.path.join(os.path.dirname(__file__), 'smplx_verts_segmentation.json')
    with open(seg_path, 'r') as f:
        smplx_segmentation = json.load(f)

    removed_body_parts = ['rightUpLeg', 'leftLeg', 'leftToeBase', 'rightFoot', 'rightLeg', 'rightToeBase', 'leftUpLeg', 'leftFoot', 'hips']
    upperbody_parts = {1,2,3,5,6,10,14,15,19,21,22,23,24,25,26,27}
    upper_body_mask = np.isin(binary_mask, list(upperbody_parts)).astype(np.uint8)
    
    removed_vertices = []
    for body_part, vertices_indices in smplx_segmentation.items():
        if body_part in removed_body_parts:
            removed_vertices.extend(vertices_indices)

    # image_height, image_width = binary_mask.shape

    # for vert_idx in smplx_segmentation['hips']:
    #     u, v = vertices_ndc[vert_idx].astype(np.int64)

    #     # Check if the projection is within the image bounds and if the mask is True
    #     if 0 <= v < image_height and 0 <= u < image_width and not upper_body_mask[v, u]:
    #         removed_vertices.append(vert_idx)

    return removed_vertices

def check_repro_error(keypoints3d, kpts_repro, keypoints2d, P, MAX_REPRO_ERROR):
    square_diff = (keypoints2d[:, :, :2] - kpts_repro[:, :, :2])**2 
    conf = keypoints3d[None, :, -1:]
    conf = (keypoints3d[None, :, -1:] > 0) * (keypoints2d[:, :, -1:] > 0)
    dist = np.sqrt((((kpts_repro[..., :2] - keypoints2d[..., :2])*conf)**2).sum(axis=-1))
    vv, jj = np.where(dist > MAX_REPRO_ERROR)
    if vv.shape[0] > 0:
        keypoints2d[vv, jj, -1] = 0.
        keypoints3d, kpts_repro = simple_recon_person(keypoints2d, P)
    return keypoints3d, kpts_repro

def mv1pmf_skel(dataset, check_repro=True, args=None):
    MIN_CONF_THRES = args.thres2d
    no_img = not (args.vis_det or args.vis_repro)
    dataset.no_img = no_img
    kp3ds = []
    start, end = args.start, min(args.end, len(dataset))
    kpts_repro = None
    for nf in tqdm(range(start, end), desc='triangulation'):
        images, annots = dataset[nf]
        check_keypoints(annots['keypoints'], WEIGHT_DEBUFF=1, min_conf=MIN_CONF_THRES)
        # print(annots['keypoints'].shape)
        keypoints3d, kpts_repro = simple_recon_person(annots['keypoints'], dataset.Pall)
        if check_repro:
            keypoints3d, kpts_repro = check_repro_error(keypoints3d, kpts_repro, annots['keypoints'], P=dataset.Pall, MAX_REPRO_ERROR=args.MAX_REPRO_ERROR)
        # keypoints3d, kpts_repro = robust_triangulate(annots['keypoints'], dataset.Pall, config=config, ret_repro=True)
        kp3ds.append(keypoints3d)
        if args.vis_det:
            dataset.vis_detections(images, annots, nf, sub_vis=args.sub_vis)
        if args.vis_repro:
            dataset.vis_repro(images, kpts_repro, nf=nf, sub_vis=args.sub_vis)
    # smooth the skeleton
    if args.smooth3d > 0:
        kp3ds = smooth_skeleton(kp3ds, args.smooth3d)
    for nf in tqdm(range(len(kp3ds)), desc='dump'):
        dataset.write_keypoints3d(kp3ds[nf], nf+start)

def mv1pmf_smpl(dataset, args, weight_pose=None, weight_shape=None):
    removed_vertices = []
    dataset.skel_path = args.skel
    kp3ds = []
    start, end = args.start, min(args.end, len(dataset))
    keypoints2d, bboxes, params_out = [], [], []
    mesh_translation = []
    dataset.no_img = True
    for nf in tqdm(range(start, end), desc='loading'):
        images, annots = dataset[nf]
        keypoints2d.append(annots['keypoints'])
        bboxes.append(annots['bbox'])
    kp3ds = dataset.read_skeleton(start, end)
    keypoints2d = np.stack(keypoints2d)
    bboxes = np.stack(bboxes)
    kp3ds = check_keypoints(kp3ds, 1)
    # optimize the human shape
    with Timer('Loading {}, {}'.format(args.model, args.gender), not args.verbose):
        body_model = load_model(gender=args.gender, model_type=args.model)
    params = smpl_from_keypoints3d2d(body_model, kp3ds, keypoints2d, bboxes, 
        dataset.Pall, config=dataset.config, args=args,
        weight_shape=weight_shape, weight_pose=weight_pose)
    # write out the results
    dataset.no_img = not (args.vis_smpl or args.vis_repro)
    for nf in tqdm(range(start, end), desc='render'):
        images, annots = dataset[nf]
        param = select_nf(params, nf-start)

        if args.flame_path is not None:
            flame_param_path = os.path.join(args.flame_path, '{:05d}.npz'.format(nf))
            flame_param = dict(np.load(flame_param_path))
            for key in flame_param.keys():
                flame_param[key] = torch.from_numpy(flame_param[key])

            mano_param_path = os.path.join(args.mano_path, '008_{:05d}.npz'.format(nf))
            mano_params = np.load(mano_param_path, allow_pickle=True)
            rmano_params = mano_params['righthand'].tolist()
            lmano_params = mano_params['lefthand'].tolist()
            for key in rmano_params.keys():
                rmano_params[key] = torch.from_numpy(rmano_params[key])

            # for key in lmano_params.keys():
            #     lmano_params[key] = torch.from_numpy(lmano_params[key])

            param['jaw_pose'] = matrix_to_euler_angles(rotation_6d_to_matrix(flame_param['jaw_pose']), convention='XYZ')
            param['reye_pose'] = matrix_to_euler_angles(rotation_6d_to_matrix(flame_param['eyes_pose'][:,:6]), convention='XYZ')
            param['leye_pose'] = matrix_to_euler_angles(rotation_6d_to_matrix(flame_param['eyes_pose'][:,6:]), convention='XYZ')
            param['expression'] = flame_param['expr']
            # param['transl'] = flame_param['translation']
            # param['global_orient'] = flame_param['rotation']
            param['right_hand_pose'] = matrix_to_euler_angles(rmano_params['hand_pose'], convention='XYZ').unsqueeze(0)
            param['left_hand_pose'] = matrix_to_euler_angles(lmano_params['hand_pose'], convention='XYZ').unsqueeze(0)
        
        params_out.append(param)
        mesh_translation.append(param['transl'])
        smplx_params_path = join(args.out, 'smplx_params')
        os.makedirs(smplx_params_path, exist_ok=True)
        np.savez(join(smplx_params_path ,f'{nf:05d}.npz'), **param)
        # dataset.write_smpl(param, nf)

        if args.write_smpl_full:
            param_full = param.copy()
            param_full['poses'] = body_model.full_poses(param['poses'])
            dataset.write_smpl(param_full, nf, mode='smpl_full')
        if args.write_vertices:
            vertices = body_model(return_verts=True, return_tensor=False, **param)
            write_data = [{'id': 0, 'vertices': vertices[0]}]
            mesh_folder = join(args.out, 'mesh')
            # dataset.write_vertices(write_data, nf)
            os.makedirs(mesh_folder, exist_ok=True)
            trimesh.Trimesh(faces=body_model.faces, vertices=vertices[0], process=False).export(f'{mesh_folder}/{nf:06d}.obj')
        if args.vis_smpl: 
            vertices = body_model(return_verts=True, return_tensor=False, **param)
            vertices_ndc = projectN3(vertices[0], Pall=dataset.Pall)
            # if nf == 0:
            #     seg_mask_path = join(args.seg_mask, '007_{:05d}_seg.npy'.format(nf))
            #     seg_mask = np.load(seg_mask_path, allow_pickle=True)
            #     removed_vertices = find_vertices_to_remove(vertices_ndc[6], seg_mask)
            #     verts, faces = remove_vertices(vertices[0], body_model.faces, removed_vertices)

            #     trimesh.Trimesh(faces=faces.cpu().numpy(), vertices=verts[0].cpu().numpy(), process=False).export(f'{mesh_folder}/{nf:06d}_upperbody.obj')
        
            dataset.vis_smpl(vertices=vertices[0], faces=body_model.faces, images=images, nf=nf, sub_vis=args.sub_vis, add_back=True)
            # render_out = dataset.pytorch3d_renderer.render_shape(vertices=vertices, images=images[:15], white=False)
            # render_out_file = join(args.out, 'render', f'{nf:06d}.png')
            # os.makedirs(join(args.out, 'render'), exist_ok=True)
            # cv2.imwrite(render_out_file, render_out)
        if args.vis_repro:
            keypoints = body_model(return_verts=False, return_tensor=False, **param)[0]
            kpts_repro = projectN3(keypoints, dataset.Pall)
            dataset.vis_repro(images, kpts_repro, nf=nf, sub_vis=args.sub_vis, mode='repro_smpl')

    np.savez(join(args.out, 'params.npz'), **{'params': params_out})
    with open(join(args.out, 'vertices.txt'), 'wb') as f:
        removed_vertices = np.array(list(removed_vertices))
        np.savetxt(f, removed_vertices, fmt='%d')

def write_canonical_smplx_param(params, tgt_folder):
    smplx_param = {
        'transl': np.zeros_like(params['transl']),
        'global_orient': np.zeros_like(params['global_orient']),
        'jaw_pose': np.array([[0.3, 0, 0]]),  # open mouth
        'lyes_pose': np.zeros_like(params['leye_pose']),
        'reye_pose': np.zeros_like(params['reye_pose']),
        'shape': params['shapes'],
        'expression': np.zeros_like(params['expression']),
        'left_hand_pose': np.zeros_like(params['left_hand_pose']),
        'right_hand_pose': np.zeros_like(params['right_hand_pose']),
        'body_pose': np.zeros_like(params['body_pose']),
    }
    
    cano_smplx_param_path = tgt_folder  + '/canonical_smplx_param.npz'
    print(f"Writing canonical SMPLX parameters to: {cano_smplx_param_path}")
    np.savez(cano_smplx_param_path, **smplx_param)

def mv1pmf_output(dataset, args):
    params = np.load(join(args.out, 'params.npz'), allow_pickle=True)['params']
    write_canonical_smplx_param(params[0], args.out)

    # for nf, item in tqdm(enumerate(params), total=len(params), desc='smplx_params'):
    #     # item['transl'] = (M[:3, 3] + item['transl']).numpy()
    #     smplx_params_path = join(args.out, 'smplx_params_new')
    #     os.makedirs(smplx_params_path, exist_ok=True)
    #     for key, val in item.items():
    #         if isinstance(val, torch.Tensor):
    #             item[key] = val.numpy()
    #     np.savez(join(smplx_params_path ,f'{nf:05d}.npz'), **item)
    dataset.write()

if __name__ == "__main__":
    from easymocap.mytools import load_parser, parse_parser
    from easymocap.dataset import CONFIG, MV1PMF
    from pytorch3d_renderer import Pytorch3DRenderer
    parser = load_parser()
    parser.add_argument('--skel', action='store_true')
    parser.add_argument('--mesh_root', type=str, default=None)
    parser.add_argument('--flame_path', type=str, default=None)
    parser.add_argument('--mano_path', type=str, default=None)
    parser.add_argument('--seg_mask', type=str, default=None)
    args = parse_parser(parser)
    help="""
  Demo code for multiple views and one person:

    - Input : {} => {}
    - Output: {}
    - Body  : {}=>{}, {}
""".format(args.path, ', '.join(args.sub), args.out, 
    args.model, args.gender, args.body)
    print(help)
    skel_path = join(args.out, 'keypoints3d')
    dataset = MV1PMF(args.path, annot_root=args.annot, cams=args.sub, out=args.out,
        config=CONFIG[args.body], kpts_type=args.body,
        undis=args.undis, no_img=False, verbose=args.verbose, mano_path=args.mano_path)
    dataset.writer.save_origin = args.save_origin

    # pytorch3d renderer
    config = {
        'image_size': [1920,1080],
        'root_folder': args.path,
    }
    dataset.pytorch3d_renderer = Pytorch3DRenderer(config)

    if args.skel or not os.path.exists(skel_path):
        mv1pmf_skel(dataset, check_repro=True, args=args)
    mv1pmf_smpl(dataset, args)
    mv1pmf_output(dataset, args)
    