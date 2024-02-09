import copy
from tkinter import Image
import open3d as open3d
from matplotlib.image import imread
import numpy as np
from scipy.spatial.transform import Rotation as R
import sys, os
import pandas as pd

def triangle_pcd(r,t,path='./data/frames/train_img4.jpg',):

    vert=[[-0.25,0.25,0],[-0.25,-0.25,0],[0.25,-0.25,0],[0.25,0.25,0]]

    faces=[[0,1,2],[0,2,3]]

    t=open3d.geometry.TriangleMesh(open3d.utility.Vector3dVector(vert), open3d.utility.Vector3iVector(faces))
    
    text=imread(path)
    img = open3d.geometry.Image(text)

    v_uv = np.array([[1,0],[0,0],[0,1],[1,0],[0,1],[1,1]])
    t.triangle_uvs = open3d.utility.Vector2dVector(v_uv)
    t.triangle_material_ids = open3d.utility.IntVector([0]*len(faces))
    t.textures=[img]
    tt = t.create_coordinate_frame()
    
    R = np.array([[-1.0000000,  0.0000000,  0.00000000], [0.0000000, 1.0000000,  0.0000000], [0.0000000,  0.0000000,  -1.0000000]])
    t.rotate(R, center=(0, 0, 0))
    s = np.array([[0, 1,  0.00000000], [-1.0000000, 0.0000000,  0.0000000], [0.0000000,  0.0000000,  1.0000000]])
    t.rotate(s, center=(0, 0, 0))

    line_vert=[[0,0,-0.5],[-0.25,0.25,0],[-0.25,-0.25,0],[0.25,-0.25,0],[0.25,0.25,0]]
    lines = [[0, 1], [0, 2], [0, 3],[0,4]]  # Right leg
    colors = [[0, 1, 0] for i in range(len(lines))]  # Default green
    line_pcd = open3d.geometry.LineSet()
    
    line_pcd.lines = open3d.utility.Vector2iVector(lines)
    line_pcd.colors = open3d.utility.Vector3dVector(colors)
    line_pcd.points = open3d.utility.Vector3dVector(line_vert)
    t.translate((0, 0, 0.5))
    line_pcd.translate((0,0,0.5))
    
    
    return t ,line_pcd, tt


def load_pointcloud(points3D_df):

    xyz = np.vstack(points3D_df['XYZ'])
    rgb = np.vstack(points3D_df['RGB'])/255

    pcd = open3d.geometry.PointCloud()
    pcd.points = open3d.utility.Vector3dVector(xyz)
    pcd.colors = open3d.utility.Vector3dVector(rgb)
    
    return pcd

def cam_to_world(R, T):

    inv_R = np.transpose(R)
    T = np.reshape(T, (3,1))
    inv_T = -1 * np.matmul(inv_R , T)
    
    return inv_R, inv_T.reshape((1,3))

def transform_matrix(rotation, translation, scale):
    r_mat = R.from_euler('xyz', rotation, degrees=True).as_matrix()
    scale_mat = np.eye(3) * scale
    transform_mat = np.concatenate([scale_mat @ r_mat, translation.reshape(3, 1)], axis=1)
    return transform_mat


if __name__ == '__main__':
    images_df = pd.read_pickle("data/images.pkl")
    valid_set = images_df.loc[ images_df["NAME"].str.contains("valid")]
    visualize = open3d.visualization.VisualizerWithKeyCallback()
    visualize.create_window()
    # load pointcloud
    points3D_df = pd.read_pickle("data/points3D.pkl")
    pcd = load_pointcloud(points3D_df)
    visualize.add_geometry(pcd)
    
    cubePointsRotate = np.load('rot.npy')
    cubePointsTrans = np.load('trans.npy')
    for index in range(len(valid_set)):
        image_name = valid_set.iloc[index][["NAME"]].values[0]
        rotq_gt = cubePointsRotate[index][0]
        tvec_gt = cubePointsTrans[index]
        t ,line,tt = triangle_pcd(rotq_gt,tvec_gt, './data/frames/' + image_name)
        r ,tr=  cam_to_world(R.from_quat(rotq_gt).as_matrix(),tvec_gt )
        
        t.translate((tr[0]))
        t.rotate(r)
        
        line.translate((tr[0]))
        line.rotate(r)
        s = np.array([[0, 1,  0.00000000], \
                [-1.0000000, 0.0000000,  0.0000000], \
                [0.0000000,  0.0000000,  1.0000000]])
        visualize.add_geometry(t)
        visualize.add_geometry(line)
    
    R_euler = np.array([0, 0, 0]).astype(float)
    t = np.array([0, 0, 0]).astype(float)
    scale = 1.0
    # initialize camera view
    vc = visualize.get_view_control()
    vc_cam = vc.convert_to_pinhole_camera_parameters()
    initial_cam = transform_matrix(np.array([10, -20, -18]), np.array([-1, 1, 5]), 1)
    initial_cam = np.concatenate([initial_cam, np.zeros([1, 4])], 0)
    initial_cam[-1, -1] = 1.
    setattr(vc_cam, 'extrinsic', initial_cam)
    vc.convert_from_pinhole_camera_parameters(vc_cam)
    shift_pressed = False
    visualize.run()
    visualize.destroy_window()