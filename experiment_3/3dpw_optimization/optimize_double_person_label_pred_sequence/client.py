from mprpc import RPCClient
import zlib
import numpy as np

client = RPCClient('127.0.0.1', 6000)


def register_test():

    opt = {
            'exp_name': 'test',
            'image_scale': 0.25,
            'img_dir_list': ['H:/paper/code/mhmr-v2/experiment_3/3dpw_optimization/optimize_double_person_label_pred_sequence/output/img/img_00',
                             'H:/paper/code/mhmr-v2/experiment_3/3dpw_optimization/optimize_double_person_label_pred_sequence/output/img/img_01'],
            'obj_dir_list': ['H:/paper/code/mhmr-v2/experiment_3/3dpw_optimization/optimize_double_person_label_pred_sequence/output/obj/obj_00',
                             'H:/paper/code/mhmr-v2/experiment_3/3dpw_optimization/optimize_double_person_label_pred_sequence/output/obj/obj_01']
          }
    render = {
        'width': {'data': zlib.compress(np.array(256).astype(np.float64).tostring()), 'shape': (1)},
        'height':  {'data': zlib.compress(np.array(256).astype(np.float64).tostring()), 'shape': (1)},
        'intrinsic':  {'data': zlib.compress(np.eye(3).astype(np.float64).tostring()), 'shape': (3,3)},
        'camera_pose':  {'data':zlib.compress(np.eye(4).astype(np.float64).tostring()), 'shape': (4,4)},
    }
    gt = {
        'img': {'data': zlib.compress(np.ones((2, 256, 256, 3)).astype(np.float64).tostring()), 'shape': (2, 256, 256, 3)},
        'mask': {'data': zlib.compress(np.ones((2, 256, 256)).astype(np.float64).tostring()), 'shape': (2, 256, 256)},
        'kp2d': {'data': zlib.compress(np.ones((2, 2, 17, 3)).astype(np.float64).tostring()), 'shape': (2, 2, 17, 3)},
    }

    r = client.call('register', opt, gt, render)

    print('register', r)



def update_test():

    exp_name = 'test'
    step_id = 200
    pred_dict = {
        'mask': {'data': zlib.compress(np.ones((2, 256, 256)).astype(np.float64).tostring()), 'shape': (2, 256, 256)},
        'kp2d': {'data': zlib.compress(np.ones((2, 2, 17, 3)).astype(np.float64).tostring()), 'shape': (2, 2, 17, 3)},
        'vertices': {'data': zlib.compress(np.array([1,2,3,4,5,6,7,8,9,1,2,3,4,5,6,7,8,9]).astype(np.float64).tostring()),
                     'shape': (2, 3, 3)},
        'faces': {'data': zlib.compress(np.array((0, 1, 2, 0, 1, 2)).astype(np.float64).tostring()),'shape': (2, 1, 3)},
    }

    r = client.call('update', exp_name, step_id, pred_dict)

    print('update', r)



if __name__ == '__main__':
    register_test()
    for i in range(100):
        print(i)
        update_test()

