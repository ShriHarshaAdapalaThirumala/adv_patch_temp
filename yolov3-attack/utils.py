
import numpy as np

#tranform vehicle coordinate to LiDAR coordinate
def trans_v2world_nuscs(added_points, gt3Dboxes, vehicle_idx,N_add):
    added_points = np.reshape(added_points,(N_add,4))
    trans = np.zeros((N_add,4), dtype=np.float32)
    w, h, l, y, z, x, yaw = gt3Dboxes[vehicle_idx]
    #print(w, h, l, y, z, x, yaw)
    trans[:, 0] = x + added_points[:,0] * np.cos(yaw) - added_points[:,1] * np.sin(yaw)
    trans[:, 1] = y + added_points[:,0] * np.sin(yaw) + added_points[:,1] * np.cos(yaw)
    trans[:, 2] = z + h/2. + added_points[:,2]
    trans[:, 3] = added_points[:,3]
    return trans