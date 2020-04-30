# 实现voxel滤波，并加载数据集中的文件进行验证

import open3d as o3d 
import os
import numpy as np
from pyntcloud import PyntCloud
import random

# 功能：对点云进行voxel滤波
# 输入：
#     point_cloud：输入点云
#     leaf_size: voxel尺寸
def voxel_filter(point_cloud, leaf_size):
    filtered_points = []
    # 作业3
    # 屏蔽开始
    datas = np.array(point_cloud)
    max_vector = datas.max(axis=0)
    min_vector = datas.min(axis = 0)
 
    arrange = max_vector - min_vector
    voxel_D = arrange / leaf_size
    point_x, point_y, point_z = np.array(point_cloud.x),np.array(point_cloud.y), np.array(point_cloud.z)
    index_x = np.floor((point_x-min_vector[0])/ leaf_size)
    index_y = np.floor((point_y-min_vector[1])/ leaf_size)
    index_z = np.floor((point_z-min_vector[2])/ leaf_size)
    index = np.floor(index_x + index_y *voxel_D[0] + index_z*voxel_D[0] *voxel_D[1])
    index_count = np.array(index)
    data = np.c_[index_count, point_x, point_y, point_z]
    data = data[data[:,0].argsort()]
 
    mode = 'hashtable'
    if mode == 'random':
        for i in range(data.shape[0]-1):
            if data[i][0] != data[i+1][0]:
                filtered_points.append(data[i][1:])
        
        filtered_points.append(data[data.shape[0]-1][1:])

    if mode == 'centroid':
        data_points = []
        for i in range(data.shape[0]-1):
            if data[i][0] == data[i+1][0]:
                data_points.append(data[i][1:])
                continue
            if data_points == []:
                continue

            filtered_points.append(np.mean(data_points,axis =0))
            data_points = []

    if mode == 'hashtable':
        voxel_D = np.ceil(voxel_D)
        voxel =[[] for _ in range(int(voxel_D[0]*voxel_D[1]*voxel_D[2]))]
        for i, point in enumerate(datas):
            index_xyz = (point-min_vector)/ leaf_size
            index_xyz = np.floor(index_xyz)
            index = round(index_xyz[0]+index_xyz[1]*voxel_D[0] + index_xyz[2]*voxel_D[0]*voxel_D[1])

            voxel[int(index)].append(i)
        func = 'random'
        if  func == 'centroid':
            for v in voxel:
                if len(v) == 0:
                    continue
                mean_data = np.sum(datas[v], axis=0)/len(v)
                filtered_points.append(mean_data)   

        if  func == 'random':
            for v in voxel:
                if len(v) == 0:
                    continue
                choice_data = datas[random.choice(v)]
                filtered_points.append(choice_data)

    # 屏蔽结束

    # 把点云格式改成array，并对外返回
    filtered_points = np.array(filtered_points, dtype=np.float64)
    return filtered_points


def voxel_filter_kint(point_cloud, leaf_size):
    filtered_points = []
    # 作业3
    # 屏蔽开始
    datas = np.array(point_cloud)
    max_vector = datas.max(axis=0)
    min_vector = datas.min(axis = 0)
 
    arrange = max_vector - min_vector
    voxel_D = arrange / leaf_size
    voxel_D = np.ceil(voxel_D).astype(np.int32)
    #filtered_points.reverse
    voxel_map = np.full(((voxel_D[0]*voxel_D[1]*voxel_D[2]).astype(np.int32), 1),-1, dtype=int)

    for data in datas:
        data_ref_min = data - min_vector
        voxel_data_index_3d = np.floor(data_ref_min / leaf_size).astype(np.int32)
        voxel_map_index = voxel_data_index_3d[0] + voxel_data_index_3d[1]*voxel_D[0] + voxel_data_index_3d[2]*voxel_D[0]*voxel_D[1]

        if voxel_map[voxel_map_index][0] == -1:
            filtered_points.append(data)
            voxel_map[voxel_map_index][0] = len(filtered_points)-1
        else:
            filtered_points_index = voxel_map[voxel_map_index][0]
            sum = data + filtered_points[filtered_points_index]
            filtered_points[voxel_map[voxel_map_index][0]] = sum/2

    # 屏蔽结束

    # 把点云格式改成array，并对外返回
    filtered_points = np.array(filtered_points, dtype=np.float64)
    return filtered_points


def main(): 
    # 指定点云路径
    dir_index = 0 # 物体编号，范围是0-39，即对应数据集中40个物体/home/yhzhao/dataset/3D/3d_pcl/modelnet40_normal_resampled_ply/airplane/airplane_0002.ply
    root_dir = '/home/yhzhao/dataset/3D/3d_pcl/modelnet40_normal_resampled_ply/' # 数据集路径
    file_dirs = os.listdir(root_dir)

    for fname in file_dirs:
        filename = os.path.join(root_dir, fname, fname+'_0001.ply') # 默认使用第一个点云
        print(filename)
        process(filename)
        

def process(file_name): 
    # 加载自己的点云文件
    #file_name = "/Users/renqian/Downloads/program/cloud_data/11.ply"
    #file_name = "/home/yhzhao/dataset/3D/3d_pcl/modelnet40_normal_resampled_ply/airplane/airplane_0001.ply"
    point_cloud_pynt = PyntCloud.from_file(file_name)

    # 转成open3d能识别的格式
    point_cloud_o3d = point_cloud_pynt.to_instance("open3d", mesh=False)
    # o3d.visualization.draw_geometries([point_cloud_o3d]) # 显示原始点云

    # 调用voxel滤波函数，实现滤波
    #filtered_cloud = voxel_filter_kint(point_cloud_pynt.points, 0.05)
    filtered_cloud = voxel_filter(point_cloud_pynt.points,  0.05)
    point_cloud_o3d.points = o3d.utility.Vector3dVector(filtered_cloud)
    # 显示滤波后的点云
    o3d.visualization.draw_geometries([point_cloud_o3d])

if __name__ == '__main__':
    main()
