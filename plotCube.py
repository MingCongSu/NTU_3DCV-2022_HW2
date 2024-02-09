import numpy as np
import os
from PIL import Image
import pandas as pd
from scipy.spatial.transform import Rotation as R
import cv2 


cameraMatrix = np.array([[1868.27,0,540],[0,1869.18,960],[0,0,1]])

def cam_to_world(R, T):

    inv_R = np.transpose(R)
    T = np.reshape(T, (3,1))
    inv_T = -1 * np.matmul(inv_R , T)
    
    return inv_R, inv_T.reshape((1,3))

def sort_by_depth(cameraPose, cubePoints):
    key = {}    
    for index in range(len(cubePoints)):
        key[index] = np.linalg.norm(cubePoints[index][0:3] - cameraPose)
    key = sorted(key.items(), key=lambda x: x[1], reverse=True)
    sort_indice = []
    
    for i in range(len(cubePoints)):
        sort_indice.append(key[i][0])

    return sort_indice
    
cubePoints = np.load('test.npy')
images_df = pd.read_pickle("data/images.pkl")
valid_set = images_df.loc[ images_df["NAME"].str.contains("valid")]
for index in range(len(valid_set)):
        image_id = valid_set.iloc[index]["IMAGE_ID"]
        save_path = "./data/img_plotCube/" + valid_set.iloc[index]["NAME"]
        print(save_path)
        img = Image.open("./data/frames/" + valid_set.iloc[index]["NAME"])
        pixel = img.load()
        
        rotq_gt = valid_set.iloc[index][["QX","QY","QZ","QW"]].values
        tvec_gt = valid_set.iloc[index][["TX","TY","TZ"]].values
        r_gt = R.from_quat(rotq_gt).as_matrix()
        t = tvec_gt
        r ,t = cam_to_world(r_gt, tvec_gt)
        r_gt = np.column_stack( (r_gt, np.transpose(tvec_gt)))
        sort_indice = sort_by_depth(t, cubePoints)
        
        for index in sort_indice:
            cubePos = cubePoints[:, 0:3]
            cubeCol = cubePoints[: , 3:]
            
            temp = np.matmul(r_gt, np.append(cubePos[index], 1))
            
            uv = np.matmul(cameraMatrix, temp)
            uv = uv / uv[2]
            if uv[0] >= 0 and uv[0] < img.size[0] and uv[1]>=0 and uv[1] <img.size[1]:
                pixel[int(uv[0]),int(uv[1])] = (int(cubeCol[index][0]), int(cubeCol[index][1]) ,int(cubeCol[index][2]))                
                try:                    
                    pixel[int(uv[0]) + 1 ,int(uv[1]) ] = (int(cubeCol[index][0]), int(cubeCol[index][1]) ,int(cubeCol[index][2]))
                    pixel[int(uv[0])  ,int(uv[1]) + 1] = (int(cubeCol[index][0]), int(cubeCol[index][1]) ,int(cubeCol[index][2]))
                    pixel[int(uv[0]) - 1 ,int(uv[1])] = (int(cubeCol[index][0]), int(cubeCol[index][1]) ,int(cubeCol[index][2]))
                    pixel[int(uv[0])  ,int(uv[1]) - 1] = (int(cubeCol[index][0]), int(cubeCol[index][1]) ,int(cubeCol[index][2]))
                except:
                    pass
                
        img.save(save_path)