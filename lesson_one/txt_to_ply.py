import os
import numpy as np
from plyfile import PlyData
from plyfile import PlyElement

# 功能：把点云信息写入ply文件，只写points
# 输入：
#     pc:点云信息
#     filename:文件名
def export_ply(pc, filename):
    vertex = np.zeros(pc.shape[0], dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4')])
    for i in range(pc.shape[0]):
        vertex[i] = (pc[i][0], pc[i][1],pc[i][2])
    ply_out = PlyData([PlyElement.describe(vertex, 'vertex', comments=['vertices'])])
    ply_filename = filename[:-4] + '.ply'
    ply_out.write(ply_filename)

# 功能：从txt文件中读取点云信息
# 输入：
#     filename:txt文件名
def read_txt(filename):
    points = []
    data = np.genfromtxt(filename,delimiter=",",dtype=float)
    points = data[:,0:3]
    #points.append([float(x) for x in value])
    points = np.array(points)
    return points

# 功能：把ModelNet数据集文件从.txt格式改成.ply格式，只包含points
# 输入：
#     ply_data_dir: ply文件的存放路径
#     txt_data_dir: txt文件的存放地址

def write_ply_points_only_from_txt(ply_data_dir, txt_data_dir):

    filelist_path = os.path.join(txt_data_dir,'filelist.txt')
    filelist = np.genfromtxt(filelist_path, delimiter='/', dtype=str)
    filelist_folder = filelist[:, 0]
    filelist_name = filelist[:, 1]
    #以和txt一样的目录结构创建输出ply的目录
    for i in range(len(filelist_folder)):
        if not os.path.exists(os.path.join(ply_data_dir, filelist_folder[i])):
            os.makedirs(os.path.join(ply_data_dir, filelist_folder[i]))
    #读取txt，输出为ply
    for i in range(len(filelist_name)):
        points = read_txt(os.path.join(txt_data_dir, filelist_folder[i], filelist_name[i]))
        name, extension = os.path.splitext(filelist_name[i])  # 分割文件名里面的名字以及后缀
        name = name+'.ply'
        export_ply(points, os.path.join(ply_data_dir, filelist_folder[i], name))


def main():
    # ply目标文件产生路径
    ply_data_dir = '/home/yhzhao/dataset/3D/3d_pcl/modelnet40_normal_resampled_ply'
    # txt文件所在路径
    txt_data_dir = '/home/yhzhao/dataset/3D/3d_pcl/modelnet40_normal_resampled'

    write_ply_points_only_from_txt(ply_data_dir, txt_data_dir)

if __name__ == '__main__':
    main()