import os
import glob
import random
import json

import cv2
from tqdm import tqdm

from lib.cfg import args, aruco_dict, charuco_board
from lib.spatial_transform import *
from lib.ur5_kinematics import UR5Kinematics as UR5Kin
import lib.cmd_printer as cmd_printer
from scipy.spatial.transform import Rotation as R

ur5_kin = UR5Kin()


class HandEyeCalibration:
    def __init__(self):
        # offset from the ee reference frame to the calibration board
        self.cv_X_r = np.array([[0, -1, 0, 0], 
                                [0, 0, -1, 0],
                                [1, 0, 0, 0],
                                [0, 0, 0, 1]])
        self.r_X_cv = np.linalg.inv(self.cv_X_r)
    
    def start(self):
        root_dir = os.path.join('./dataset', args.dataset)
        calib_path = os.path.join(root_dir, 'calib.json')
        assert os.path.exists(calib_path), "'calib.json' does not exist"
        calib_dict = json.load(open(calib_path, 'r'))
        meta = calib_dict['meta']
        
        for camera, mode in meta.items():
            cmd_printer.divider(text='Calibrating <{}>'.format(camera))
            path_temp = os.path.join(root_dir, camera)
            imdb = self.preprocess_images(path_temp)
            assert os.path.exists(path_temp), '{} does not exist'.format(path_temp)
            if args.load_intrinsics: 
                try:
                    intrinsics = calib_dict[camera]['intrinsics']
                except KeyError:
                    intrinsics = self.intrinsics_calibration(imdb)
            else:
                intrinsics = self.intrinsics_calibration(imdb)
            calib_dict[camera] = {}
            calib_dict[camera]['intrinsics'] = intrinsics
            calib_dict[camera]['extrinsics'] = self.calibrate(
                imdb, intrinsics, mode)
        # TODO: provide an option to save as euler angles
        json.dump(calib_dict, open(calib_path, 'w'), indent=4)
        print('\nSystem calibration file saved to:\n {}'.format(calib_path))

    def calibrate(self, imdb, intrinsics=None, mode=None):
        n_samples = len(imdb.keys())
        if imdb:
            # Estimating camera pose w.r.t Robot Base...
            # TODO: Estiamte Table Height
            C_X_M = np.zeros((4, 4, n_samples))
            X_H = np.zeros((4, 4, n_samples))
            for i, img_id in enumerate(imdb.keys()):
                C_X_M[..., i] = self.cmpt_C_X_M(
                    imdb[img_id]["corners"],
                    imdb[img_id]["ids"],
                    intrinsics,
                    convention=args.convention)
                X_H[..., i]= ur5_kin.get_ee_pose(
                    np.array(imdb[img_id]["joint_config"]).reshape(6, 1))
            rot, trans = self.hand_eye_calibration(X_H, C_X_M, mode)
            cmd_printer.divider(
                'Camera Extrinsics in <{}> convention'.format(args.convention),
                char='-')
            rt = np.eye(4)
            rt[:3,:3], rt[:3, 3] = rot, trans.flatten()
            print(rt)
            return {'R': rot.tolist(), 't':trans.tolist()}
        else:
            print('IMDB is empty')
            return None

    def hand_eye_calibration(self, X_H, Ccv_X_M, mode):
        assert mode in ('eye_in_hand', 'external'), \
            'Undefined calibration mode'
        n = X_H.shape[2]
        assert n > 4, 'No enough sample points'  
        rand_idx = np.arange(n)
        # random.shuffle(rand_idx)
        k = int(n/2)
        A, B = np.zeros((4, 4, k)), np.zeros((4, 4, k))
        for ctr, i in enumerate(range(0, n-1, 2)):
            id_0, id_1 = rand_idx[i], rand_idx[i+1]
            X_H0, X_H1 = X_H[:, :, id_0], X_H[:, :, id_1]
            Ccv_X_M0, Ccv_X_M1 = Ccv_X_M[..., id_0], Ccv_X_M[..., id_1]
            if mode == 'eye_in_hand':
                A[..., ctr] = np.linalg.inv(X_H1).dot(X_H0)
            elif mode == 'external':
                A[..., ctr] = X_H1.dot(np.linalg.inv(X_H0))
            B[..., ctr] = Ccv_X_M1.dot(np.linalg.inv(Ccv_X_M0))
        rot, trans = self.solve_ax_xb(A, B)
        return rot, trans

    @staticmethod
    def calibrate_plane(X_Ccv, Ccv_X_M):
        # do rotation averaging:
        n = X_Ccv.shape[2]
        assert n > 2, 'Not enough samples'
        X_M = np.zeros((n, 4, 4))
        for i in range(n):
            X_M[i, :, :] = X_Ccv[:, :, i].dot(Ccv_X_M[:, :, i])
        X_M_avg, err_mean, err_std = SE3_avg(X_M)
        print('rotation avg error: {}+\-{}'.format(err_mean, err_std))
        return X_M_avg
    
    @staticmethod
    def solve_ax_xb(A, B):
        _, _, n = A.shape
        A_ = np.zeros((9*n, 9))
        b = np.zeros((9*n, 1))
        for i in range(n):
            Ra = A[:3, :3, i]
            Rb = B[:3, :3, i]
            A_[9*i: 9*(i+1), :] = np.kron(Ra, np.eye(3)) +\
                np.kron(-np.eye(3), Rb.T)
        _, _, vh = np.linalg.svd(A_)
        v = vh.T
        x = v[:, -1]
        R = x.reshape(3, 3)
        det_R = np.linalg.det(R)
        R = np.sign(det_R)/abs(det_R)**(1/3.0) * R
        u, s, vh = np.linalg.svd(R)
        R = u.dot(vh)
        if np.linalg.det(R) < 0:
            R = u .dot(np.diag([1, 1, -1])).dot(vh)
        C = np.zeros((3*n, 3))
        d = np.zeros((3*n, 1))
        I = np.eye(3)
        for i in range(n):
            C[3*i: 3*(i+1), :] = I - A[:3, :3, i]
            d[3*i: 3*(i+1), 0] = A[:3, 3, i] -  R.dot(B[:3, 3, i])
        t = np.linalg.lstsq(C, d, rcond=None)[0]
        return R, t

    def preprocess_images(self, dataset_dir):
        image_dir = os.path.join(dataset_dir, 'images')
        meta_dir = os.path.join(dataset_dir, 'meta')
        all_meta_paths = glob.glob(os.path.join(meta_dir, '*.json'))
        data_dict = {}
        print('Extracting All Image Corners:')
        for meta_path in tqdm(all_meta_paths, ncols=59):
            meta = json.load(open(meta_path))
            image_name = meta['image_name']
            cv2_img = cv2.imread(os.path.join(image_dir, image_name))
            corners_temp, ids_temp = self.get_corners(cv2_img)
            if corners_temp is not None:
                temp_dict = {"corners": corners_temp, "ids": ids_temp,
                             "joint_config": meta["joint_config"],
                             "im_size": cv2_img.shape}
                data_dict[image_name] = temp_dict
            else:
                print('No enough corners detected, skip %s' % image_name)
        return data_dict

    @staticmethod
    def intrinsics_calibration(data_dict):
        print("Estimating camera intrinsics...")
        all_image_names = data_dict.keys()
        all_corners, all_ids = [], []
        for image_name in all_image_names:
            all_corners.append(data_dict[image_name]["corners"])
            all_ids.append(data_dict[image_name]["ids"])
            im_size = data_dict[image_name]["im_size"][:2]
        try:
            _, cam_mat, dist_coeffs, _, _ = cv2.aruco.calibrateCameraCharuco(
                charucoCorners=all_corners, charucoIds=all_ids,
                board=charuco_board, imageSize=im_size,
                cameraMatrix=None, distCoeffs=None)
            print("K:\n {}".format(np.around(cam_mat[0:3], decimals=4)))
            print("Distortion coefficients:\n {}".format(np.around(
                dist_coeffs[0], decimals=3)))
            return {"K": cam_mat.tolist(), "distortion": dist_coeffs.tolist()}
        except:
            print("Intrinsic calibration failed. Recalibrating...")

    def cmpt_C_X_M(self, corners, ids, intrinsics, convention='robotic'):
        retval, rvecs, tvecs = cv2.aruco.estimatePoseCharucoBoard(
            corners, ids, charuco_board, np.array(intrinsics["K"]),
            np.array(intrinsics["distortion"]))
        if retval:
            Ccv_X_M = np.eye(4)
            Ccv_X_M[:3, 3] = tvecs[:3].reshape(3)
            Ccv_X_M[0:3, 0:3] = cv2.Rodrigues(rvecs[:])[0]
            if convention == 'robotics':
                C_X_M = self.r_X_cv.dot(Ccv_X_M)
            elif convention == 'cv':
                C_X_M = Ccv_X_M
            # print(C_X_M)
            return C_X_M
        else:
            print('Board Pose not detected')
            return None

    @staticmethod
    def get_corners(cv2_img):
        gray_img = cv2.cvtColor(cv2_img, cv2.COLOR_BGR2GRAY)
        corners, ids, _ = cv2.aruco.detectMarkers(image=gray_img,
                                              dictionary=aruco_dict)
        if ids is not None and len(ids) > 5:
            _, charuco_corners, charuco_ids = cv2.aruco.interpolateCornersCharuco(
                corners, ids, gray_img, charuco_board)
            return charuco_corners, charuco_ids
        else:
            print("Too few markers detected, skip...")
            return None, None


if __name__ == "__main__":
    cmd_printer.divider(text="Arguments", line_max=60)
    for arg in vars(args):
        print('{}: {}'.format(arg, getattr(args, arg)))
    
    calib = HandEyeCalibration()
    calib.start()
