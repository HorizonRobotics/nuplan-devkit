import numpy as np

def get_transform_matrix(translation, rotation, forward):
    t_mat = np.eye(4)
    t_mat[:3, -1] = translation
    r_mat = np.eye(4)
    r_mat[:3, :3] = rotation.rotation_matrix

    if forward:
        # first translate points, then rotate points
        return r_mat @ t_mat
    # first rotate points, thentranslate points
    return t_mat @ r_mat


def rescale_K(K_, s, keep_fov=True):
    K = K_.copy()
    K[0, 2] = s[0] * K[0, 2]
    K[1, 2] = s[1] * K[1, 2]
    if keep_fov:
        K[0, 0] = s[0] * K[0, 0]
        K[1, 1] = s[1] * K[1, 1]
    return K