from scipy.spatial.transform import Rotation as R
from sympy import * 
from cv2 import solveP3P
import pandas as pd
import numpy as np
import cv2
import math
import statistics
from icp import icp
import time

images_df = pd.read_pickle("data/images.pkl")
train_df = pd.read_pickle("data/train.pkl")
points3D_df = pd.read_pickle("data/points3D.pkl")
point_desc_df = pd.read_pickle("data/point_desc.pkl")

x_ = Symbol('x_', real=True)
d1_ = Symbol('dd1', real=True, positive=True)

def average(x):
    return list(np.mean(x,axis=0))

def average_desc(train_df, points3D_df):
    train_df = train_df[["POINT_ID","XYZ","RGB","DESCRIPTORS"]]
    desc = train_df.groupby("POINT_ID")["DESCRIPTORS"].apply(np.vstack)
    desc = desc.apply(average)
    desc = desc.reset_index()
    desc = desc.join(points3D_df.set_index("POINT_ID"), on="POINT_ID")
    return desc

def undistortion(pixel, distCoeff):
    
    x = pixel[0] / pixel[2]
    y = pixel[1] / pixel[2]
    r2 = x*x +y*y
    f = 1 + distCoeff[0] * r2 + distCoeff[1] * r2 * r2
    x = x * f
    y = y * f
    
    dx = x + 2 * distCoeff[2] * x * y + distCoeff[3]* (r2 + 2*x*x) 
    dy = y + 2 * distCoeff[3] * x * y + distCoeff[2]*(r2 + 2*y*y) 
    x = dx
    y = dy 
    
    return np.array([x,y,1])

def compute_all_loss(rotq, rotq_gt, tvec, tvec_gt):
    rotation_matrix_pred = R.from_quat(rotq).as_matrix()[0]
    assert(rotation_matrix_pred.shape == (3,3))
    rotation_matrix_gt = R.from_quat(rotq_gt).as_matrix()
    rotation_matrix_error = np.matmul(rotation_matrix_pred, np.linalg.inv(rotation_matrix_gt))
    angle_error = math.acos((rotation_matrix_error[0][0] + rotation_matrix_error[1][1]  + rotation_matrix_error[2][2]  - 1 ) / 2)
    
    translation_error = np.linalg.norm(tvec - tvec_gt)
        
    return angle_error, translation_error

def compute_G(k1, k2, cos_thetas):

    G4 = (k1 * k2 - k1 - k2 ) ** 2 - 4 * k1 * k2 * cos_thetas[1] * cos_thetas[1]
    
    G3 = 4 * (k1 * k2 - k1 - k2 ) * k2 * (1 - k1) * cos_thetas[0] + 4 * k1 * cos_thetas[1] * ((k1 * k2 - k1 + k2) * cos_thetas[2] + 2 * k2 * cos_thetas[0] * cos_thetas[1])
    
    G2 = (2 * k2 * (1 - k1) * cos_thetas[0]) ** 2 + 2 * (k1 * k2 - k1 - k2 ) * (k1 * k2 + k1 - k2) + 4 * k1 * ( (k1 - k2) * cos_thetas[1] * cos_thetas[1] + k1 * (1 - k2) * cos_thetas[2] * cos_thetas[2] - 2 * (1 + k1) * k2 * cos_thetas[0] * cos_thetas[1] * cos_thetas[2])
    
    G1 = 4 * (k1 * k2 + k1 - k2) * k2 * (1 - k1) * cos_thetas[0] + 4 * k1 * ((k1 * k2 - k1 + k2) * cos_thetas[2] * cos_thetas[1] + 2 * k1 * k2 * cos_thetas[0] * cos_thetas[2] * cos_thetas[2])
    
    G0 = (k1 * k2 + k1 - k2) ** 2 - 4 * k1 *k1 * k2 * cos_thetas[2] * cos_thetas[2]  
        
    return [G4,G3,G2,G1,G0]

def cos_theta(x1,x2,k):

    k_inv = np.linalg.inv(k)
    k_tran_inv = np.linalg.inv( np.transpose(k))
    
    IAC = np.matmul(k_tran_inv, k_inv)
    
    numerator = np.matmul(x1, np.matmul(IAC, x2.reshape(3,1)))
    Denominator = math.sqrt( np.matmul(x1, np.matmul(IAC, x1.reshape(3,1)))  *  np.matmul(x2, np.matmul(IAC, x2.reshape(3,1))) )

    return numerator / Denominator

def p3p_ransac(points3D,points2D, cameraMatrix, distCoeffs ,rotq_gt,tvec_gt):
    # 設定ransac參數
    max_step = 200
    camaraMatrix_inv = np.linalg.inv(cameraMatrix)
    points2D = np.append(points2D, np.ones((points2D.shape[0],1)),axis=1)
    distance = 1
    res = []
    min_r_loss = 10000000

    for i in range(max_step):
        # 隨機sample 3個點
        index_random_sample_point = np.random.choice(len(points2D), 3)
        random_sample_point_2d = points2D[index_random_sample_point]
        random_sample_point_3d = points3D[index_random_sample_point]       
        # 計算sample出的3點在世界座標中的距離
        distance = np.array([np.linalg.norm(random_sample_point_3d[0] - random_sample_point_3d[1]),\
            np.linalg.norm(random_sample_point_3d[1] - random_sample_point_3d[2]),\
            np.linalg.norm(random_sample_point_3d[0] - random_sample_point_3d[2])])
        # 計算每個cosine theta
        cos_thetas = np.array([cos_theta(random_sample_point_2d[0], random_sample_point_2d[1], cameraMatrix), \
                    cos_theta(random_sample_point_2d[1], random_sample_point_2d[2], cameraMatrix),\
                    cos_theta(random_sample_point_2d[0], random_sample_point_2d[2], cameraMatrix)]).reshape(-1,3)[0]
        # 依照上課ppt之求解p3p方法，利用餘弦定理求解四次多項式之解x
        k1 = (distance[1] / distance[2]) ** 2
        k2 = (distance[1] / distance[0]) ** 2
        G = compute_G(k1,k2,cos_thetas)
        x = solve(G[4] * (x_**4) + G[3] * (x_**3) + G[2] * (x_**2) + G[1] * (x_) + G[0], x_)
        # 利用內參的inverse和sample點計算到世界座標的單位向量
        o1 = np.matmul(camaraMatrix_inv, np.transpose(random_sample_point_2d[0])) 
        o2 = np.matmul(camaraMatrix_inv, np.transpose(random_sample_point_2d[1]))
        o3 = np.matmul(camaraMatrix_inv, np.transpose(random_sample_point_2d[2]))
        o1 = o1 / np.linalg.norm(o1)
        o2 = o2 / np.linalg.norm(o2)
        o3 = o3 / np.linalg.norm(o3)
        # 驗證每組解
        for sol in x:
            # 嘗試於每個解出的x中找出正確的解
            y = -(sol ** 2 - k1 - (k1 -1) * (sol**2 * (k2 - 1) - 2 * k2 * sol * cos_thetas[0] + k2)  ) / (2 * k1 * (cos_thetas[2] - sol * cos_thetas[1]))
            d1 = solve((1 + sol**2 - 2 * sol * cos_thetas[0]) * d1_ ** 2 - distance[0] ** 2, d1_)[0]
            d2 = d1 * sol
            d3 = d1 * y
            O1_temp = o1 * float(d1)
            O2_temp = o2 * float(d2)
            O3_temp = o3 * float(d3) 
            
            norm_vec = np.cross((O2_temp - O1_temp), (O3_temp - O1_temp))    
            # 利用icp求旋轉平移
            c_o = np.array([O1_temp,O2_temp,O3_temp])
            T, _ ,_= icp(random_sample_point_3d[0:3], c_o)
            T= T[0:3, :]
            # 利用得到的外在參數計算影像平面上的誤差，以利ransac運作
            l2_dis = 0
            for i in range(len(points2D)):
                a = points2D[i]
                temp = np.matmul(T, np.transpose(np.append(points3D[i],1)))
                temp = undistortion(temp, distCoeffs)
                b = np.matmul(cameraMatrix, np.transpose(temp))
                b = b / b[2]
                l2_dis +=np.linalg.norm(a-b)
            
            if l2_dis <min_r_loss:
                res = T
                min_r_loss = l2_dis
    # 回傳外在參數           
    rotM = R.from_matrix(res[0:3, 0:3]).as_rotvec()
    return rotM, res[0:3, 3]

def pnpsolver(query,model,rotq_gt,tvec_gt):
    kp_query, desc_query = query
    kp_model, desc_model = model

    bf = cv2.BFMatcher()
    matches = bf.knnMatch(desc_query,desc_model,k=2)

    gmatches = []
    for m,n in matches:
        if m.distance < 0.75*n.distance:
            gmatches.append(m)

    points2D = np.empty((0,2))
    points3D = np.empty((0,3))

    for mat in gmatches:
        query_idx = mat.queryIdx
        model_idx = mat.trainIdx
        points2D = np.vstack((points2D,kp_query[query_idx]))
        points3D = np.vstack((points3D,kp_model[model_idx]))

    cameraMatrix = np.array([[1868.27,0,540],[0,1869.18,960],[0,0,1]])    
    distCoeffs = np.array([0.0847023,-0.192929,-0.000201144,-0.000725352])

    r, t = p3p_ransac(points3D, points2D, cameraMatrix, distCoeffs,rotq_gt,tvec_gt)
    return 1,r,t,2

def opencv_pnpsolver(query,model,cameraMatrix=0,distortion=0):
    kp_query, desc_query = query
    kp_model, desc_model = model

    bf = cv2.BFMatcher()
    matches = bf.knnMatch(desc_query,desc_model,k=2)

    gmatches = []
    for m,n in matches:
        if m.distance < 0.75*n.distance:
            gmatches.append(m)

    points2D = np.empty((0,2))
    points3D = np.empty((0,3))

    for mat in gmatches:
        query_idx = mat.queryIdx
        model_idx = mat.trainIdx
        points2D = np.vstack((points2D,kp_query[query_idx]))
        points3D = np.vstack((points3D,kp_model[model_idx]))

    cameraMatrix = np.array([[1868.27,0,540],[0,1869.18,960],[0,0,1]])    
    distCoeffs = np.array([0.0847023,-0.192929,-0.000201144,-0.000725352])

    return cv2.solvePnPRansac(points3D, points2D, cameraMatrix, distCoeffs,flags=solveP3P )

if __name__ == '__main__':
    # Process model descriptors
    desc_df = average_desc(train_df, points3D_df)
    kp_model = np.array(desc_df["XYZ"].to_list())
    desc_model = np.array(desc_df["DESCRIPTORS"].to_list()).astype(np.float32)
    # Load validation set
    valid_set = images_df.loc[ images_df["NAME"].str.contains("valid")]
    
    angle_errors = []
    translation_errors = []
    rot_preds = []
    t_preds = []

    for index in range(len(valid_set)):
        image_id = valid_set.iloc[index]["IMAGE_ID"]
        # Load query keypoints and descriptors
        points = point_desc_df.loc[point_desc_df["IMAGE_ID"]== image_id]
        kp_query = np.array(points["XY"].to_list())
        desc_query = np.array(points["DESCRIPTORS"].to_list()).astype(np.float32)
        #load groundtruth
        rotq_gt = valid_set.iloc[index][["QX","QY","QZ","QW"]].values
        tvec_gt = valid_set.iloc[index][["TX","TY","TZ"]].values
        # Find correspondance and solve pnp
        retval, rvec, tvec, inliers = pnpsolver((kp_query, desc_query),(kp_model, desc_model),rotq_gt,tvec_gt)
        rotq = R.from_rotvec(rvec.reshape(1,3)).as_quat()
        tvec = tvec.reshape(1,3)
        # predicit
        rot_preds.append(rotq)
        t_preds.append(tvec)
        # records errors
        angle_error, translation_error = compute_all_loss(rotq,rotq_gt,tvec,tvec_gt)
        angle_errors.append(angle_error)
        translation_errors.append(translation_error)
        
    # print error
    print(statistics.median(angle_errors))
    print(statistics.median(translation_errors))
    # save rotation and translation
    np.save('./rot.npy', np.asarray(rot_preds))
    np.save('./trans.npy', np.asarray(t_preds))
