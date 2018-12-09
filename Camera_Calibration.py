# -*- coding: utf-8 -*-
# """
# Created on Sat Nov 24 16:49:45 2018
# 
# @author: Carlos Isaac
# """
import numpy as np
import cv2
import glob
from matplotlib import pyplot as plt
import pathlib

def Camera_Calibration():
    # termination criteria
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
    objp = np.zeros((6*9,3), np.float32)
    objp[:,:2] = np.mgrid[0:6,0:9].T.reshape(-1,2)

    # Arrays to store object points and image points from all the images.
    objpoints = [] # 3d point in real world space
    imgpoints = [] # 2d points in image plane.

    images = glob.glob('./calibration_img/*.JPG')

    for fname in images:
        img = cv2.imread(fname)
        gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        #img = cv2.resize(img, (0,0), fx=0.3, fy=0.3) 
        # Find the chess board corners
        ret, corners = cv2.findChessboardCorners(gray, (6,9),None)
#        print("Detected corners in picture "+fname+": "+str(ret))
        # If found, add object points, image points (after refining them)
        if ret == True:
            objpoints.append(objp)
            
            cv2.cornerSubPix(gray,corners,(11,11),(-1,-1),criteria)
            imgpoints.append(corners)
            
            # Draw and display the corners
            cv2.drawChessboardCorners(img, (9,6), corners,ret)
            cv2.imwrite('./calibration_img/calibration1.JPG',img)
            cv2.imshow('img',img)
            cv2.waitKey(500)
            
        cv2.destroyAllWindows()

    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1],None,None)

    print("\nFinal re-projection error: "+str(ret)+"\n")
    print("Camera Matrix: ")    
    print(mtx)
    print('\nVector of distortion coefficients:')
    print(dist)
    return  ret, mtx, dist

def undistortion():
    count = 0
    images = glob.glob('./scene_2/*.JPG')
    for fname in images:
        #    fname_2 = fname 
        img = cv2.imread(fname)
        #    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        img_u = cv2.undistort(img, mtx, dist)
        outfname = "./scene_2/" + str(count)+".JPG"
        check = pathlib.Path("./scene_2/0.JPG")
        #print(check.is_file())
        if not check.is_file():
            cv2.imwrite(str(outfname),img_u)
        count = count + 1


def drawlines(img1,img2,lines,pts1,pts2):
    ''' img1 - image on which we draw the epilines for the points in img2
        lines - corresponding epilines '''
    r,c = img1.shape
    img1 = cv2.cvtColor(img1,cv2.COLOR_GRAY2BGR)
    img2 = cv2.cvtColor(img2,cv2.COLOR_GRAY2BGR)
    for r,pt1,pt2 in zip(lines,pts1,pts2):
        color = tuple(np.random.randint(0,255,3).tolist())
        x0,y0 = map(int, [0, -r[2]/r[1] ])
        x1,y1 = map(int, [c, -(r[2]+r[0]*c)/r[1] ])
        img1 = cv2.line(img1, (x0,y0), (x1,y1), color,1)
        img1 = cv2.circle(img1,tuple(pt1),5,color,-1)
        img2 = cv2.circle(img2,tuple(pt2),5,color,-1)
    return img1,img2


ret, mtx, dist = Camera_Calibration()    
undistortion()

#
#img = cv2.imread('./scene_1/1.JPG')
#gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
#sift = cv2.xfeatures2d.SIFT_create()
#kp1 = sift.detectAndCompute(img,None)
#
#img = cv2.imread('./scene_1/2.JPG')
#gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
#sift = cv2.xfeatures2d.SIFT_create()
#kp2 = sift.detectAndCompute(img,None)


img1 = cv2.imread('./scene_2/0.JPG') # queryImage
img2 = cv2.imread('./scene_2/3.JPG') # trainImage
img1 = cv2.cvtColor(img1,cv2.COLOR_BGR2GRAY)
img2 = cv2.cvtColor(img2,cv2.COLOR_BGR2GRAY)

# Initiate SIFT detector
sift = cv2.xfeatures2d.SIFT_create()
# find the keypoints and descriptors with SIFT
kp1, des1 = sift.detectAndCompute(img1,None)
kp2, des2 = sift.detectAndCompute(img2,None)

# FLANN parameters
FLANN_INDEX_KDTREE = 0
index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
search_params = dict(checks=50)

flann = cv2.FlannBasedMatcher(index_params,search_params)
matches = flann.knnMatch(des1,des2,k=2)

good = []
pts1 = []
pts2 = []

# ratio test as per Lowe's paper
for i,(m,n) in enumerate(matches):
    if m.distance < 0.5*n.distance:
        good.append(m)
        pts2.append(kp2[m.trainIdx].pt)
        pts1.append(kp1[m.queryIdx].pt)
        
pts1 = np.int32(pts1)
pts2 = np.int32(pts2)
F, mask = cv2.findFundamentalMat(pts1,pts2,cv2.FM_LMEDS)

# We select only inlier points
pts1 = pts1[mask.ravel()==1]
pts2 = pts2[mask.ravel()==1]


# Find epilines corresponding to points in right image (second image) and
# drawing its lines on left image
lines1 = cv2.computeCorrespondEpilines(pts2.reshape(-1,1,2), 2,F)
lines1 = lines1.reshape(-1,3)
img5,img6 = drawlines(img1,img2,lines1,pts1,pts2)

# Find epilines corresponding to points in left image (first image) and
# drawing its lines on right image
lines2 = cv2.computeCorrespondEpilines(pts1.reshape(-1,1,2), 1,F)
lines2 = lines2.reshape(-1,3)
img3,img4 = drawlines(img2,img1,lines2,pts2,pts1)

plt.subplot(121),plt.imshow(img5)
plt.subplot(122),plt.imshow(img3)
plt.show()

cv2.imwrite('epi1.jpg',img5)
cv2.imwrite('epi2.jpg',img3)

focal = mtx[1][1]
principalPoint = (mtx[0][2],mtx[1][2])

#Mat, E = cv2.findEssentialMat(pts1, pts2, focal, principalPoint, method = cv2.RANSAC, 0.999, 3, mask= noArray() );

E, mask = cv2.findEssentialMat(pts1, pts2, focal, principalPoint, method=cv2.RANSAC, prob=0.999, threshold=3.0)

R1,R2,t = cv2.decomposeEssentialMat(E)

origin = np.float64([0,0,0])
origin = origin[:, np.newaxis]

Project1 = np.concatenate((np.eye(3, dtype='float64'), origin),1)
Project1 = mtx @ Project1
Project2_1 = np.concatenate((R1, t),1)
Project2_1 = mtx @ Project2_1
Project2_2 = np.concatenate((R1, -t),1)
Project2_2 = mtx @ Project2_2
Project2_3 = np.concatenate((R2, t),1)
Project2_3 = mtx @ Project2_3
Project2_4 = np.concatenate((R2, -t),1)
Project2_4 = mtx @ Project2_4

points2d_1_aux0 = np.array(pts1)
points2d_2_aux0 = np.array(pts2)

points2d_1_aux = points2d_1_aux0[1500:1600]
points2d_2_aux = points2d_2_aux0[1500:1600]

points2d_1 = points2d_1_aux.astype(float).transpose()
points2d_2 = points2d_2_aux.astype(float).transpose()


#pts1_test = np.array([[pts1[0][0]],[pts1[0][1]]])
#pts2_test = np.array([[pts2[0][0]],[pts2[0][1]]])


P_Triang1 = cv2.triangulatePoints(Project1, Project2_1, points2d_1, points2d_2)
P_Triang2 = cv2.triangulatePoints(Project1, Project2_2, points2d_1, points2d_2)
P_Triang3 = cv2.triangulatePoints(Project1, Project2_3, points2d_1, points2d_2)
P_Triang4 = cv2.triangulatePoints(Project1, Project2_4, points2d_1, points2d_2)

P_T_3D = cv2.convertPointsFromHomogeneous(P_Triang3.transpose())

P_T_2D = cv2.convertPointsFromHomogeneous(P_T_3D)

P_T_2D = cv2.convertPointsToHomogeneous(P_T_2D)

PixelP = P_T_2D[:,0,:].T

PixelP = mtx @ PixelP

PixelP = cv2.convertPointsFromHomogeneous(PixelP)


#for point in P_T_3D:
#    projected_PC = point
#    projected_PC = projected_PC /projected_PC[0][2]
#    projected_PCn = mtx @ projected_PC  
#    print(projected_PC)


#[rvec, J] = cv2.Rodrigues(R1)
#
#projected_P = cv2.projectPoints(P_T_3D, rvec, t, mtx, dist, J)


#P4d_test = [[P4d[0][0]],[P4d[1][0]],[P4d[2][0]], [P4d[3][0]]]
#P4d_test = P4d_test/P4d_test[3][0]
#P4d_test1 = (P4d_test[0][0],P4d_test[1][0],P4d_test[2][0])

img1 = cv2.imread('./scene_2/0.JPG')


    
#for point in P_T_H:
#    projected_PC = Pose1 @ point.transpose()
#    projected_PCn = mtx @ projected_PC
    
  

#    img1 = cv2.circle(img1,tuple(projected_P),5,(0, 255, 0),-1)
    
#Print_P_T_2D = P_T_2D.astype(int)
   
#P_3D_Pixel = np.true_divide(P_T_3D[:2,:], P_T_3D[:,[-1]])
    
for point in points2d_1.astype(int).transpose():
    img1 = cv2.circle(img1,tuple(point),5,(0, 255, 0),-1)

for point in PixelP.astype(int):
    img1 = cv2.circle(img1,tuple(point[0]),5,(255, 0, 0),-1)

img1 = cv2.resize(img1, (0,0), fx=0.3, fy=0.3) 

cv2.imshow('img',img1)
cv2.waitKey(0)
cv2.destroyAllWindows()

R1v = cv2.Rodrigues(R1)
R1v = R1v[0]

#Proj_Points = cv2.projectPoints(P4d_test1,R1v,t,mtx,dist)


    
    
    
    
    