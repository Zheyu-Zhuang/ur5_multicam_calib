import cv2
import glob
import json
import os

import cv2.aruco as aruco
from tqdm import tqdm

from lib.cfg import args, aruco_dict, charuco_board
from lib.spatial_transform import *
from lib.ur5_kinematics import UR5Kinematics as UR5Kin

ur5_kin = UR5Kin()


class MVC:
    def __init__(self):
        self.M_X_E = np.array([[0, -1, 0, 0.23],
                               [-1, 0, 0, 0.19],
                               [0, 0, -1, -0.011],
                               [0, 0, 0, 1]])
        # offset between end-effector and table hen board is on table
        self.board_height_offset = np.eye(4)
        self.board_height_offset[2, 3] = -0.027
        # offset from the ee reference frame to the calibration board
        self.E_X_T = np.matmul(np.linalg.inv(self.M_X_E),
                               self.board_height_offset)
        self.cv2robotics = np.array([[0, -1, 0, 0], [0, 0, -1, 0],
                                     [1, 0, 0, 0], [0, 0, 0, 1]])

    def start(self):
        print("\n#################")
        print("Start calibrating")
        print("#################")
        root, sub_dirs, _ = os.walk(
            os.path.join('./dataset', args.dataset)).next()
        calib_file_path = os.path.join(root, 'system_calib.json')
        calib_dict = {}
        if os.path.exists(calib_file_path) and args.load_intrinsics:
            calib_dict = json.load(open(calib_file_path, 'r'))
        for dir_temp in sub_dirs:
            print('\nCalibrating {}'.format(dir_temp))
            path_temp = os.path.join(root, dir_temp)
            if args.load_intrinsics and dir_temp in calib_dict:
                calib_dict[dir_temp] = self.calibrate(path_temp,
                                                      calib_dict[dir_temp][
                                                          "intrinsics"])
            else:
                calib_dict[dir_temp] = self.calibrate(path_temp)
        json.dump(calib_dict, open(calib_file_path, 'w'))
        print('System calibration file saved to:\n {}'.format(calib_file_path))

    def calibrate(self, dataset_dir, intrinsics=None):
        data_dict = self.preprocess_images(dataset_dir)
        output_dict = {}
        if data_dict:
            all_image_names = data_dict.keys()
            n_images = len(all_image_names)
            if intrinsics is None:
                all_corners, all_ids = [], []
                for image_name in all_image_names:
                    all_corners.append(data_dict[image_name]["corners"])
                    all_ids.append(data_dict[image_name]["ids"])
                    im_size = data_dict[image_name]["im_size"][:2]
                print("Estimating camera intrinsics...")
                intrinsics = self.get_camera_intrinsics(
                    all_corners, all_ids, im_size)
            output_dict['intrinsics'] = intrinsics
            # Estimating camera pose w.r.t Robot Base...
            B_X_C_stack = np.zeros((n_images, 4, 4))
            for i, image_name in enumerate(all_image_names):
                corners_temp = data_dict[image_name]["corners"]
                ids_temp = data_dict[image_name]["ids"]
                joint_config = data_dict[image_name]["joint_config"]
                B_X_C_stack[i] = self.get_camera_pose_from_sample(
                    corners_temp, ids_temp,
                    np.array(joint_config).reshape(6, 1), intrinsics)
            avg_cam_pose, err_mean, err_std = SE3_avg(B_X_C_stack)
            print("Pose of camera w.r.t Base:\n{}".format(
                np.around(avg_cam_pose, decimals=4)))
            print("L2 Norm Error Mean {}+/-{}\n".format(round(err_mean, 4),
                                                     round(err_std, 4)))
            output_dict['pose'] = avg_cam_pose.tolist()
            _, sub_dirs, _ = os.walk(dataset_dir).next()
            table_dir_idx = [i for i, x in enumerate(sub_dirs) if 'tab' in x]
            for idx_temp in table_dir_idx:
                print('Estimating table height...')
                table_id = sub_dirs[idx_temp]
                dataset_dir = os.path.join(dataset_dir, table_id)
                data_dict = self.preprocess_images(dataset_dir)
                all_image_names = data_dict.keys()
                table_height = 0
                err_stat = []
                for i, image_name in enumerate(all_image_names):
                    corners_temp = data_dict[image_name]["corners"]
                    ids_temp = data_dict[image_name]["ids"]
                    joint_config = data_dict[image_name]["joint_config"]
                    height_temp, err = self.get_table_height_from_sample(
                        corners_temp, ids_temp, avg_cam_pose,
                        np.array(joint_config).reshape(6, 1), intrinsics)
                    if height_temp is not None:
                        table_height += height_temp
                        err_stat.append(err)
                avg_table_height = table_height / len(data_dict.keys())
                output_dict["{}_height".format(table_id)] = avg_table_height
                err_mean = round(np.mean(err_stat), 4)
                err_std = round(np.std(err_stat), 4)
                print("{} height w.r.t the base is {}".format(
                    table_id, round(avg_table_height, 4)))
                print("Error between vision and kin estimations: {}+/-{}".format(
                    err_mean, err_std))
            return output_dict

    def preprocess_images(self, dataset_dir):
        image_dir = os.path.join(dataset_dir, 'images')
        meta_dir = os.path.join(dataset_dir, 'meta')
        all_meta_paths = glob.glob(os.path.join(meta_dir, '*.json'))
        data_dict = {}
        print("Extracting corners from all images")
        for meta_path in tqdm(all_meta_paths):
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
    def get_camera_intrinsics(all_corners, all_ids, im_size):
        try:
            _, cam_mat, dist_coeffs, _, _ = aruco.calibrateCameraCharuco(
                charucoCorners=all_corners, charucoIds=all_ids,
                board=charuco_board, imageSize=im_size,
                cameraMatrix=None, distCoeffs=None)
            print("K:\n {}".format(np.around(cam_mat[0:3], decimals=4)))
            print("Distortion coefficients:\n {}".format(np.around(
                dist_coeffs[0], decimals=3)))
            return {"K": cam_mat.tolist(), "distortion": dist_coeffs.tolist()}
        except:
            print("Intrinsic calibration failed. Recalibrating...")

    def get_ee_wrt_cam(self, corners, ids, intrinsics, convention='robotic'):
        retval, rvecs, tvecs = aruco.estimatePoseCharucoBoard(
            corners, ids, charuco_board, np.array(intrinsics["K"]),
            np.array(intrinsics["distortion"]))
        if retval:
            Ccv_X_M = np.eye(4)
            Ccv_X_M[:3, 3] = tvecs[:3].reshape(3)
            Ccv_X_M[0:3, 0:3] = cv2.Rodrigues(rvecs[:])[0]
            Ccv_X_E = Ccv_X_M.dot(self.M_X_E)
            E_X_Ccv = np.linalg.inv(Ccv_X_E)
            if convention == 'robotic':
                return E_X_Ccv.dot(self.cv2robotics)
            else:
                return E_X_Ccv
        else:
            print('Board Pose not detected')
            return None

    def get_camera_pose_from_sample(self, corners, ids, joint_config,
                                    intrinsics):
        E_X_C = self.get_ee_wrt_cam(corners, ids, intrinsics)
        if E_X_C is not None:
            B_X_E = ur5_kin.get_ee_pose(joint_config)
            return B_X_E.dot(E_X_C)
        else:
            return None

    def get_table_height_from_sample(self, corners, ids, B_X_C, joint_config,
                                     intrinsics):
        E_X_C = self.get_ee_wrt_cam(corners, ids, intrinsics)
        if E_X_C is not None:
            B_X_T_vision = B_X_C.dot(np.linalg.inv(E_X_C)).dot(self.E_X_T)
            height_from_vision = B_X_T_vision[2, 3]
            B_X_E = ur5_kin.get_ee_pose(joint_config)
            B_X_T_kin = B_X_E.dot(self.E_X_T)
            height_from_kin = B_X_T_kin[2, 3]
            err = height_from_kin - height_from_vision
            return height_from_kin, err
        else:
            return None, None

    @staticmethod
    def get_corners(cv2_img):
        gray_img = cv2.cvtColor(cv2_img, cv2.COLOR_BGR2GRAY)
        corners, ids, _ = aruco.detectMarkers(image=gray_img,
                                              dictionary=aruco_dict)
        if ids is not None and len(ids) > 5:
            _, charuco_corners, charuco_ids = aruco.interpolateCornersCharuco(
                corners, ids, gray_img, charuco_board)
            return charuco_corners, charuco_ids
        else:
            print("Too few markers detected, skip...")
            return None, None


if __name__ == "__main__":
    calib = MVC()
    calib.start()
