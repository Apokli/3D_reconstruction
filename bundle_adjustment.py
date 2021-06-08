import numpy as np
from scipy.sparse import lil_matrix
from scipy.optimize import least_squares
import cv2


class PyBA:
    def __init__(self, kp3ds, Rots, trans, kps_list, d2to3ds_list, K):
        self.K = K
        self.total_params = []
        self.camera_params = []
        self.n_cameras = len(Rots)
        self.n_points = len(kp3ds)
        self.cam_indices = []
        self.point_indices = []
        self.point_2ds = []
        self.kp3ds = np.array(kp3ds)
        self.kps_list = kps_list
        self.d2to3ds_list = d2to3ds_list
        self.prepare_params(kp3ds, Rots, trans)
        self.prepare_2d_points_and_indices()

    def prepare_params(self, kp3ds, Rots, trans):
        for i in range(self.n_cameras):
            r = cv2.Rodrigues(Rots[i])
            camera_param = np.zeros((1, 6))
            camera_param[0, :3] = r[0].T
            camera_param[0, 3:6] = trans[i].T
            self.camera_params.append(camera_param)
        self.camera_params = np.array(self.camera_params).squeeze()
        self.total_params = np.hstack((self.camera_params.ravel(), self.kp3ds.ravel()))

    def prepare_2d_points_and_indices(self):
        for i, struct_marks in enumerate(self.d2to3ds_list):
            for j, mark in enumerate(struct_marks):
                if mark >= 0:
                    self.point_2ds.append(self.kps_list[i][j].pt)
                    self.cam_indices.append(i)
                    self.point_indices.append(mark)
        self.point_2ds = np.array(self.point_2ds)
        self.cam_indices = np.array(self.cam_indices, dtype=int)
        self.point_indices = np.array(self.point_indices, dtype=int)

    def compute_residual(self, params):
        """Compute residuals.

        `params` contains camera parameters and 3-D coordinates.
        """
        camera_params = params[:self.n_cameras * 6].reshape((self.n_cameras, 6))
        points_3d = params[self.n_cameras * 6:].reshape((self.n_points, 3))
        residuals = []
        for i in range(len(self.point_2ds)):
            kp3d = points_3d[self.point_indices[i]]
            kp2d = self.point_2ds[i]
            R = cv2.Rodrigues(camera_params[self.cam_indices[i], :3])
            Rot = R[0]
            tran = np.array(camera_params[self.cam_indices[i], 3:6])
            rep, jac = cv2.projectPoints(kp3d, Rot, tran, self.K, np.array([]))
            residual = rep - kp2d
            residuals.append(residual)
        return np.array(residuals).ravel()

    def bundle_adjustment_sparsity(self):
        m = self.cam_indices.size * 2
        n = self.n_cameras * 6 + self.n_points * 3
        A = lil_matrix((m, n), dtype=int)

        i = np.arange(self.cam_indices.size)
        for s in range(6):
            A[2 * i, self.cam_indices * 6 + s] = 1
            A[2 * i + 1, self.cam_indices * 6 + s] = 1

        for s in range(3):
            A[2 * i, self.n_cameras * 6 + self.point_indices * 3 + s] = 1
            A[2 * i + 1, self.n_cameras * 6 + self.point_indices * 3 + s] = 1
        return A

    def run_BA(self):
        A = self.bundle_adjustment_sparsity()
        res = least_squares(self.compute_residual, self.total_params, jac_sparsity=A, verbose=2, x_scale='jac', ftol=1e-4, method='trf')
        optimized_params = res.x
        cam_pos = optimized_params[:self.n_cameras * 6].reshape((self.n_cameras, 6))
        Rots = []
        trans = []
        for i in range(cam_pos.shape[0]):
            Rots.append(cv2.Rodrigues(cam_pos[i][:3])[0])
            trans.append(np.array([[cam_pos[i][3]], [cam_pos[i][4]], [cam_pos[i][5]]]))
        kp3ds = optimized_params[self.n_cameras * 6:].reshape((self.n_points, 3))
        return res, Rots, trans, kp3ds