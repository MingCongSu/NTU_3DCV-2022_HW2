import os
import cv2
import numpy as np
import open3d as o3d

points = []
for i in range(1,40):
    for k in range(1,40):
        points.append([0.025*i , 1 , 0.025*k , 255,0,0])
for i in range(1,40):
    for k in range(1,40):
        points.append([0, 0.025*i, 0.025*k, 200, 100, 30])        
for i in range(1,40):
    for k in range(1,40):
        points.append([0.025*i, 0, 0.025*k, 0,0,255])        
for i in range(1,40):
    for k in range(1,40):
        points.append([1, 0.025*i, 0.025*k, 255,0,127])
for i in range(1,40):
    for k in range(1,40):
        points.append([ 0.025*i, 0.025*k,1, 127,127,0])        
for i in range(1,40):
    for k in range(1,40):
        points.append([ 0.025*i, 0.025*k,0, 0,255,0])
points = np.array(points)
np.save('test.npy', np.asarray(points))