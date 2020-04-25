# -*- coding: UTF-8 -*-
# 实现PCA分析和法向量计算，并加载数据集中的文件进行验证

import open3d as o3d 
import os
import numpy as np
from pyntcloud import PyntCloud

from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d import proj3d
from matplotlib.patches import FancyArrowPatch

class Arrow3D(FancyArrowPatch):
    def __init__(self, xs, ys, zs, *args, **kwargs):
        FancyArrowPatch.__init__(self, (0,0), (0,0), *args, **kwargs)
        self._verts3d = xs, ys, zs

    def draw(self, renderer):
        xs3d, ys3d, zs3d = self._verts3d
        xs, ys, zs = proj3d.proj_transform(xs3d, ys3d, zs3d, renderer.M)
        self.set_positions((xs[0],ys[0]),(xs[1],ys[1]))
        FancyArrowPatch.draw(self, renderer)



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
    print("aver \n ", aver)
    A_aver = A - aver
    
    X_aver = A_aver.T
    print("new_A \n",  X_aver[0:3,:])

    H = np.dot(X_aver,X_aver.T)
    print("H\n",H)

    eigenvalues, eigenvectors = np.linalg.eig(H) # column eigenvectors[:,i] is the eigenvector
    print("eigenvalues", eigenvalues)
    print(" eigenvectors", eigenvectors)

    for i in range(len(eigenvectors)):
        print(eigenvectors[:,i])
        print(eigenvectors[:,i].reshape(1,3))
        eigv = eigenvectors[:,i].reshape(1,3).T
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


def main():
    # 指定点云路径
    # cat_index = 10 # 物体编号，范围是0-39，即对应数据集中40个物体
    # root_dir = '/Users/renqian/cloud_lesson/ModelNet40/ply_data_points' # 数据集路径
    # cat = os.listdir(root_dir)
    # filename = os.path.join(root_dir, cat[cat_index],'train', cat[cat_index]+'_0001.ply') # 默认使用第一个点云

    # 加载原始点云
    point_cloud_pynt = PyntCloud.from_file("/home/yhzhao/dataset/3D/3d_pcl/modelnet40_normal_resampled_ply/airplane/airplane_0001.ply")

    print(point_cloud_pynt)
    point_cloud_o3d = point_cloud_pynt.to_instance("open3d", mesh=False)
    o3d.visualization.draw_geometries([point_cloud_o3d]) # 显示原始点云

    # 从点云中获取点，只对点进行处理
    points = point_cloud_pynt.points
    #print(type(point_cloud_pynt))
    #print(points)

    print('total points number is:', points.shape[0])

    # 用PCA分析点云主方向
    w, v = PCA(points)
    point_cloud_vector = v[:, 0:3] #点云主方向对应的向量
    print('the main orientation of this pointcloud is: ', point_cloud_vector)
    # TODO: 此处只显示了点云，还没有显示PCA
    # o3d.visualization.draw_geometries([point_cloud_o3d])

    eig_vec = point_cloud_vector.T
    print(eig_vec)
    points_array = np.array(points.values).T
    #print(points_array)
    transformed = point_cloud_vector.T.dot(points_array)
    print(transformed)
    plt.plot(transformed[0,0:20], transformed[1,0:20], 'o', markersize=7, color='blue', alpha=0.5, label='class1')
    plt.plot(transformed[0,20:40], transformed[1,20:40], '^', markersize=7, color='red', alpha=0.5, label='class2')
    plt.xlim([-4,4])
    plt.ylim([-4,4])
    plt.xlabel('x_values')
    plt.ylabel('y_values')
    plt.legend()
    plt.title('Transformed samples with class labels')

    plt.show()


    # 循环计算每个点的法向量
    pcd_tree = o3d.geometry.KDTreeFlann(point_cloud_o3d)
    normals = []
    # 作业2
    # 屏蔽开始

    # 由于最近邻搜索是第二章的内容，所以此处允许直接调用open3d中的函数

    # 屏蔽结束
    normals = np.array(normals, dtype=np.float64)
    # TODO: 此处把法向量存放在了normals中
    point_cloud_o3d.normals = o3d.utility.Vector3dVector(normals)
    o3d.visualization.draw_geometries([point_cloud_o3d])


if __name__ == '__main__':
    main()
