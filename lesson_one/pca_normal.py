# -*- coding: UTF-8 -*-
# 实现PCA分析和法向量计算，并加载数据集中的文件进行验证

import open3d as o3d 
import os
import numpy as np
from pyntcloud import PyntCloud

 
# 功能：计算PCA的函数
# 输入：
#     data：点云，NX3的矩阵
#     correlation：区分np的cov和corrcoef，不输入时默认为False
#     sort: 特征值排序，排序是为了其他功能方便使用，不输入时默认为True
# 输出：
#     eigenvalues：特征值
#     eigenvectors：特征向量
def PCA(data, correlation=False, sort=True):
    # 作业1
    # 屏蔽开始
    A = np.array(data.values)
    aver = np.mean(A, axis=0)
    #print("aver \n ", aver)
    A_aver = A - aver
    
    X_aver = A_aver.T
    #print("new_A \n",  X_aver[0:3,:])

    H = np.dot(X_aver,X_aver.T)
    #print("H\n",H)

    #eigenvalues, eigenvectors = np.linalg.eig(H) # column eigenvectors[:,i] is the eigenvector
    eigenvectors, eigenvalues, eigenvectors_t = np.linalg.svd(H) # column eigenvectors[:,i] is the eigenvector
    #print("eigenvalues", eigenvalues)
    #print(" eigenvectors", eigenvectors)

    #for i in range(len(eigenvectors)):
        #print(eigenvectors[:, i])
        #print(eigenvectors[:, i].reshape(1, 3))
        #eigv = eigenvectors[:, i].reshape(1, 3).T
        # a = H.dot(eigv)
        # b = eigv * eigenvalues[i]
        # c = a-b
        # print(c)
        #np.testing.assert_array_almost_equal(H.dot(eigenvalues), eigv * eigenvalues, decimal=6, err_msg='', verbose=True)

    ### debug display
    # fig = plt.figure(figsize=(7,7))
    # ax = fig.add_subplot(111, projection='3d')

    # ax.plot(A[:,0], A[:,1], A[:,2], 'o', markersize=8, color='green', alpha=0.2)
    # ax.plot([aver[0]], [aver[1]], [aver[2]], 'o', markersize=10, color='red', alpha=0.5)
    # for v in eigenvectors.T:
    #     a = Arrow3D([aver[0], v[0]], [aver[1], v[1]], [aver[2], v[2]], mutation_scale=20, lw=3, arrowstyle="-|>", color="r")
    #     ax.add_artist(a)
    # ax.set_xlabel('x_values')
    # ax.set_ylabel('y_values')
    # ax.set_zlabel('z_values')
    # plt.title('Eigenvectors')
    # plt.show()

    # 屏蔽结束

    if sort:
        sort = eigenvalues.argsort()[::-1]
        eigenvalues = eigenvalues[sort]
        eigenvectors = eigenvectors[:, sort]

    return eigenvalues, eigenvectors

def PCA2(A, correlation=False, sort=True):
    aver = np.mean(A, axis=0)
    A_aver = A - aver
    X_aver = A_aver.T
    H = np.dot(X_aver,X_aver.T)

    #eigenvalues, eigenvectors = np.linalg.eig(H) # column eigenvectors[:,i] is the eigenvector
    eigenvectors, eigenvalues, eigenvectors_t = np.linalg.svd(H) # column eigenvectors[:,i] is the eigenvector

    #for i in range(len(eigenvectors)):
        #print(eigenvectors[:, i])
        #print(eigenvectors[:, i].reshape(1, 3))
    #    eigv = eigenvectors[:, i].reshape(1, 3).T
 
    if sort:
        sort = eigenvalues.argsort()[::-1]
        eigenvalues = eigenvalues[sort]
        eigenvectors = eigenvectors[:, sort]

    return eigenvalues, eigenvectors

def main(): 
    # 指定点云路径
    dir_index = 0 # 物体编号，范围是0-39，即对应数据集中40个物体/home/yhzhao/dataset/3D/3d_pcl/modelnet40_normal_resampled_ply/airplane/airplane_0002.ply
    root_dir = '/home/yhzhao/dataset/3D/3d_pcl/modelnet40_normal_resampled_ply/' # 数据集路径
    file_dirs = os.listdir(root_dir)

    for fname in file_dirs:
        filename = os.path.join(root_dir, fname, fname+'_0001.ply') # 默认使用第一个点云
        print(filename)
        process(filename)

def process(filename): 
    # 指定点云路径
    # cat_index = 10 # 物体编号，范围是0-39，即对应数据集中40个物体
    # root_dir = '/Users/renqian/cloud_lesson/ModelNet40/ply_data_points' # 数据集路径
    # cat = os.listdir(root_dir)
    # filename = os.path.join(root_dir, cat[cat_index],'train', cat[cat_index]+'_0001.ply') # 默认使用第一个点云

    # 加载原始点云
    #point_cloud_pynt = PyntCloud.from_file("/home/yhzhao/dataset/3D/3d_pcl/modelnet40_normal_resampled_ply/airplane/airplane_0001.ply")
    point_cloud_pynt = PyntCloud.from_file(filename)

    #print(point_cloud_pynt)
    point_cloud_o3d = point_cloud_pynt.to_instance("open3d", mesh=False)
    #o3d.visualization.draw_geometries([point_cloud_o3d])  # 显示原始点云

    # 从点云中获取点，只对点进行处理
    points = point_cloud_pynt.points
    #print('total points number is:', points.shape[0])

    # 用PCA分析点云主方向
    w, v = PCA(points)
    point_cloud_vector = v[:, 0:3]  # 点云主方向对应的向量
    #print('the main orientation of this pointcloud is: ', point_cloud_vector)
    # TODO: 此处只显示了点云，还没有显示PCA
    #o3d.visualization.draw_geometries([point_cloud_o3d])
    #print(type(point_cloud_o3d))
    #print(type([point_cloud_o3d]))
    direct_points =[[0, 0, 0], v[:,0], v[:,1]]
    lines = [[0,1] , [0,2]]
    colors =[[1,0,0], [0,1,0]]
    line_set = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector(direct_points),
        lines=o3d.utility.Vector2iVector(lines)
    )
    line_set.colors = o3d.utility.Vector3dVector(colors)
    o3d.visualization.draw_geometries([point_cloud_o3d, line_set])
    
    # test_pcd = o3d.geometry.PointCloud()
    # datas = np.array(points)
    # pca_datas = np.dot(datas, point_cloud_vector);
    # #print(pca_datas)
    # #print(type([pca_datas][0][0]))
    # test_pcd.points = o3d.utility.Vector3dVector(pca_datas)  # 定义点云坐标位置
    # #o3d.visualization.draw_geometries([test_pcd])

    # 循环计算每个点的法向量
    pcd_tree = o3d.geometry.KDTreeFlann(point_cloud_o3d)
    normals = []
    # 作业2
    # 屏蔽开始
    datas = np.array(point_cloud_pynt.points)
    for data in point_cloud_o3d.points:
        [k, idx, _] = pcd_tree.search_knn_vector_3d(data, 50)
        near_data = datas[np.array(idx)]
        near_eig_value, near_eig_vector = PCA2(points)
        direct = near_eig_vector[:, 2]#*0.2;  # 点云法方向对应的向量
        normals.append(direct)

    # 由于最近邻搜索是第二章的内容，所以此处允许直接调用open3d中的函数

    # 屏蔽结束
    normals = np.array(normals, dtype=np.float64)
    # TODO: 此处把法向量存放在了normals中
    point_cloud_o3d.normals = o3d.utility.Vector3dVector(normals)
    o3d.visualization.draw_geometries([point_cloud_o3d])


if __name__ == '__main__':
    main()
