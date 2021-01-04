import numpy as np
import pickle
import os
import sys
import cv2
import time
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

abspath = os.path.abspath(os.path.dirname(__file__))


class SMPL_np():
  def __init__(self, model_path, joint_type='smpl'):
    """
    SMPL model.

    Parameter:
    ---------
    joint_type: 'smpl' or 'cocoplus'. original smpl has 24 joints, cocoplus has 19 joints.
    model_path: Path to the SMPL model parameters, pre-processed by
    `preprocess.py`.

    """
    self.joint_type = joint_type

    with open(model_path, 'rb') as f:
      # params = pickle.load(f,encoding='iso-8859-1')
      params = pickle.load(f,encoding='iso-8859-1')

      if self.joint_type == 'cocoplus':
        self.J_cocoplus_regressor = params['cocoplus_regressor']
      self.J_regressor = params['J_regressor']
      self.weights = params['weights']
      self.posedirs = params['posedirs']
      self.v_template = params['v_template']
      self.shapedirs = params['shapedirs']
      self.faces = params['f']
      self.kintree_table = params['kintree_table']

    id_to_col = {
      self.kintree_table[1, i]: i for i in range(self.kintree_table.shape[1])
    }
    self.parent = {
      i: id_to_col[self.kintree_table[0, i]]
      for i in range(1, self.kintree_table.shape[1])
    }

    self.pose_shape = [24, 3]
    self.beta_shape = [10]
    self.trans_shape = [3]

    self.pose = np.zeros(self.pose_shape)
    self.beta = np.zeros(self.beta_shape)
    self.trans = np.zeros(self.trans_shape)

    self.verts = None
    self.J = None
    self.R = None

    self.update()

  def set_params(self, pose=None, beta=None, trans=None, v_template=None):
    """
    Set pose, shape, and/or translation parameters of SMPL model. Verices of the
    model will be updated and returned.

    Prameters:
    ---------
    pose: Also known as 'theta', a [24,3] matrix indicating child joint rotation
    relative to parent joint. For root joint it's global orientation.
    Represented in a axis-angle format.

    beta: Parameter for model shape. A vector of shape [10]. Coefficients for
    PCA component. Only 10 components were released by MPI.

    trans: Global translation of shape [3].

    Return:
    ------
    Updated vertices.

    """
    if pose is not None:
      self.pose = pose
    if beta is not None:
      self.beta = beta
    if trans is not None:
      self.trans = trans
    if v_template is not None:
      self.v_template = v_template

    self.update()
    return self.verts

  def update(self):
    """
    Called automatically when parameters are updated.

    """
    # how beta affect body shape
    v_shaped = self.shapedirs.dot(self.beta) + self.v_template
    # joints location
    self.J = self.J_regressor.dot(v_shaped)
    pose_cube = self.pose.reshape((-1, 1, 3))
    # rotation matrix for each joint
    self.R = self.rodrigues(pose_cube)
    I_cube = np.broadcast_to(
      np.expand_dims(np.eye(3), axis=0),
      (self.R.shape[0]-1, 3, 3)
    )
    lrotmin = (self.R[1:] - I_cube).ravel()
    # how pose affect body shape in zero pose
    v_posed = v_shaped + self.posedirs.dot(lrotmin)
    # world transformation of each joint
    G = np.empty((self.kintree_table.shape[1], 4, 4))
    G[0] = self.with_zeros(np.hstack((self.R[0], self.J[0, :].reshape([3, 1]))))
    for i in range(1, self.kintree_table.shape[1]):
      G[i] = G[self.parent[i]].dot(
        self.with_zeros(
          np.hstack(
            [self.R[i],((self.J[i, :]-self.J[self.parent[i],:]).reshape([3,1]))]
          )
        )
      )
    # remove the transformation due to the rest pose
    G = G - self.pack(
      np.matmul(
        G,
        np.hstack([self.J, np.zeros([24, 1])]).reshape([24, 4, 1])
        )
      )
    # transformation of each vertex
    T = np.tensordot(self.weights, G, axes=[[1], [0]])
    rest_shape_h = np.hstack((v_posed, np.ones([v_posed.shape[0], 1])))
    v = np.matmul(T, rest_shape_h.reshape([-1, 4, 1])).reshape([-1, 4])[:, :3]
    self.verts = v + self.trans.reshape([1, 3])

    # new joints, joints are gotten by current shape
    if self.joint_type == 'cocoplus':
      self.J = self.J_cocoplus_regressor.dot(self.verts)
    else:
      self.J = self.J_regressor.dot(self.verts)


  def rodrigues(self, r):
    """
    Rodrigues' rotation formula that turns axis-angle vector into rotation
    matrix in a batch-ed manner.

    Parameter:
    ----------
    r: Axis-angle rotation vector of shape [batch_size, 1, 3].

    Return:
    -------
    Rotation matrix of shape [batch_size, 3, 3].

    """
    theta = np.linalg.norm(r, axis=(1, 2), keepdims=True)
    # avoid zero divide
    theta = np.maximum(theta, np.finfo(np.float64).tiny)
    r_hat = r / theta
    cos = np.cos(theta)
    z_stick = np.zeros(theta.shape[0])
    m = np.dstack([
      z_stick, -r_hat[:, 0, 2], r_hat[:, 0, 1],
      r_hat[:, 0, 2], z_stick, -r_hat[:, 0, 0],
      -r_hat[:, 0, 1], r_hat[:, 0, 0], z_stick]
    ).reshape([-1, 3, 3])
    i_cube = np.broadcast_to(
      np.expand_dims(np.eye(3), axis=0),
      [theta.shape[0], 3, 3]
    )
    A = np.transpose(r_hat, axes=[0, 2, 1])
    B = r_hat
    dot = np.matmul(A, B)
    R = cos * i_cube + (1 - cos) * dot + np.sin(theta) * m
    return R

  def with_zeros(self, x):
    """
    Append a [0, 0, 0, 1] vector to a [3, 4] matrix.

    Parameter:
    ---------
    x: Matrix to be appended.

    Return:
    ------
    Matrix after appending of shape [4,4]

    """
    return np.vstack((x, np.array([[0.0, 0.0, 0.0, 1.0]])))

  def pack(self, x):
    """
    Append zero matrices of shape [4, 3] to vectors of [4, 1] shape in a batched
    manner.

    Parameter:
    ----------
    x: Matrices to be appended of shape [batch_size, 4, 1]

    Return:
    ------
    Matrix of shape [batch_size, 4, 4] after appending.

    """
    return np.dstack((np.zeros((x.shape[0], 4, 3)), x))

  def save_obj(self, path):
    """
    Save the SMPL model into .obj file.

    Parameter:
    ---------
    path: Path to save.

    """
    with open(path, 'w') as fp:
      for v in self.verts:
        fp.write('v %f %f %f\n' % (v[0], v[1], v[2]))
      for f in self.faces + 1:
        fp.write('f %d %d %d\n' % (f[0], f[1], f[2]))

  def get_obj(self):
    """
    get the joints, verts and faces from SMPL model.

    Return:
    ------
    Dict of joints, vertices and faces of .obj file.
    ---------
    """
    return {
      'verts': self.verts,
      'faces': self.faces,
      'J': self.J
    }

if __name__ == '__main__':
  smpl = SMPL_np(abspath + "/../data/neutral_smpl_with_cocoplus_reg.pkl", joint_type='cocoplus')
  np.random.seed(9608)

  # pose = (np.random.rand(*smpl.pose_shape) - 0.5) * 0.4
  pose = np.zeros((24, 3))
  beta = (np.random.rand(10) - 0.5) * 0.06

  smpl.set_params(beta=beta, pose=pose)

  obj = smpl.get_obj()

  vertices = obj['verts']
  faces = obj['faces']
  joints = obj['J']

  ## show
  import pyrender
  import trimesh

  # mesh
  vertex_colors = np.ones([vertices.shape[0], 4]) * [0.3, 0.3, 0.3, 0.8]
  tri_mesh = trimesh.Trimesh(vertices, faces,
                             vertex_colors=vertex_colors, )
  mesh = pyrender.Mesh.from_trimesh(tri_mesh, wireframe=True)
  scene = pyrender.Scene()
  scene.add(mesh)

  # joint
  sm = trimesh.creation.uv_sphere(radius=0.005)
  sm.visual.vertex_colors = [0.1, 0.1, 0.9, 1.0]
  tfs = np.tile(np.eye(4), (len(joints), 1, 1))
  tfs[:, :3, 3] = joints
  joints_pcl = pyrender.Mesh.from_trimesh(sm, poses=tfs)
  scene.add(joints_pcl)

  # plot_joints_index
  J_regressor_vertex = np.concatenate((vertices[smpl.J_cocoplus_regressor[7].nonzero()[1]],
                                       vertices[smpl.J_cocoplus_regressor[10].nonzero()[1]],
                                       vertices[smpl.J_cocoplus_regressor[6].nonzero()[1]],
                                       vertices[smpl.J_cocoplus_regressor[11].nonzero()[1]],
                                       vertices[smpl.J_cocoplus_regressor[2].nonzero()[1]],
                                       vertices[smpl.J_cocoplus_regressor[3].nonzero()[1]],
                                       vertices[smpl.J_cocoplus_regressor[1].nonzero()[1]],
                                       vertices[smpl.J_cocoplus_regressor[4].nonzero()[1]],
                                       vertices[smpl.J_cocoplus_regressor[0].nonzero()[1]],
                                       vertices[smpl.J_cocoplus_regressor[5].nonzero()[1]]), axis=0)
  sm = trimesh.creation.uv_sphere(radius=0.005)
  sm.visual.vertex_colors = [0.1, 0.9, 0.1, 1.0]
  tfs = np.tile(np.eye(4), (len(J_regressor_vertex), 1, 1))
  tfs[:, :3, 3] = J_regressor_vertex
  joints_pcl = pyrender.Mesh.from_trimesh(sm, poses=tfs)
  scene.add(joints_pcl)


  pyrender.Viewer(scene, use_raymond_lighting=True)
