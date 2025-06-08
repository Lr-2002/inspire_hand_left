import rerun as rr
import numpy as np
import pinocchio as pin
import os
import random
import trimesh
import hppfcl

# 设置URDF文件路径
current_dir = os.path.dirname(os.path.abspath(__file__))
urdf_file = os.path.join(current_dir, 'urdf/left.urdf')

# 初始化一个 Rerun 记录器
rr.init("urdf_visualization", spawn=True)

# 使用Pinocchio加载URDF模型
model = pin.buildModelFromUrdf(urdf_file)
data = model.createData()

# 加载几何模型
# 创建一个字典，将包名映射到实际路径
package_dirs = {
    "urdf_left": os.path.join(os.path.dirname(os.path.dirname(urdf_file)), "left")
}

# 使用简化的方法，不使用几何模型
# 我们将直接从文件系统加载网格

# 查找所有的网格文件
mesh_dir = "/home/lr-2002/project/utpt/inspire_hand/left/meshes"
mesh_files = {}
if os.path.exists(mesh_dir):
    for file in os.listdir(mesh_dir):
        if file.endswith(".STL") or file.endswith(".stl"):
            mesh_name = os.path.splitext(file)[0]
            mesh_files[mesh_name] = os.path.join(mesh_dir, file)

print(f"找到 {len(mesh_files)} 个网格文件")


# 获取模型中的所有链接
link_ids = {}
for frame_id, frame in enumerate(model.frames):
    if frame.type == pin.FrameType.BODY:
        link_name = frame.name
        link_ids[link_name] = frame_id

# 设置随机关节角度以便可视化
q = pin.randomConfiguration(model)

# 计算前向运动学
pin.forwardKinematics(model, data, q)
pin.updateFramePlacements(model, data)

# 为每个链接生成采样点并可视化
for link_name, frame_id in link_ids.items():
    # 获取链接的位置和旋转
    frame_placement = data.oMf[frame_id]
    rotation = frame_placement.rotation
    translation = frame_placement.translation
    
    # 尝试从网格文件中获取点
    all_local_points = []
    
    # 尝试匹配链接名称与网格文件名称
    link_name_lower = link_name.lower()
    matching_meshes = []
    
    # 尝试不同的命名方式进行匹配
    for mesh_name, mesh_path in mesh_files.items():
        mesh_name_lower = mesh_name.lower()
        if link_name_lower in mesh_name_lower or mesh_name_lower in link_name_lower:
            matching_meshes.append(mesh_path)
    
    # 如果没有匹配的网格，尝试分解链接名称
    if not matching_meshes and "_" in link_name_lower:
        parts = link_name_lower.split("_")
        for mesh_name, mesh_path in mesh_files.items():
            mesh_name_lower = mesh_name.lower()
            for part in parts:
                if part in mesh_name_lower and len(part) > 2:  # 避免匹配太短的字符串
                    matching_meshes.append(mesh_path)
                    break
    
    print(f"链接 {link_name} 匹配到 {len(matching_meshes)} 个网格文件")
    
    # 从匹配的网格文件中采样点
    for mesh_path in matching_meshes:
        try:
            mesh = trimesh.load(mesh_path)
            # 从网格表面采样点
            num_samples = min(500, max(100, 500 // len(matching_meshes)))  # 根据网格数量分配采样点数
            points, _ = trimesh.sample.sample_surface(mesh, num_samples)
            for p in points:
                all_local_points.append(p)
        except Exception as e:
            print(f"Error loading mesh {mesh_path}: {e}")
    
    # 如果没有成功采样到点，则使用随机点
    if not all_local_points:
        print(f"为链接 {link_name} 生成随机点")
        local_points = np.random.uniform(-0.02, 0.02, (500, 3))  # 在一个小立方体范围内生成点
    else:
        local_points = np.array(all_local_points)
        # 如果点太多，可以随机采样减少数量
        if len(local_points) > 2000:
            indices = np.random.choice(len(local_points), 2000, replace=False)
            local_points = local_points[indices]
    
    # 将点从局部坐标系转换到全局坐标系
    global_points = np.zeros((len(local_points), 3))
    for i, point in enumerate(local_points):
        global_points[i] = translation + rotation @ point
    
    # 为每个链接生成不同的颜色
    r = random.randint(100, 255)
    g = random.randint(100, 255)
    b = random.randint(100, 255)
    colors = np.ones((len(local_points), 3), dtype=np.uint8)
    colors[:, 0] = r
    colors[:, 1] = g
    colors[:, 2] = b
    
    # 向 rerun 发送点云数据
    rr.log(f"robot/{link_name}", rr.Points3D(positions=global_points, colors=colors))
    
    # 添加坐标系可视化
    axes_points = np.array([
        [0, 0, 0],
        [0.03, 0, 0],  # X轴
        [0, 0, 0],
        [0, 0.03, 0],  # Y轴
        [0, 0, 0],
        [0, 0, 0.03]   # Z轴
    ])
    
    # 将坐标轴点从局部坐标系转换到全局坐标系
    global_axes = np.zeros_like(axes_points)
    for i, point in enumerate(axes_points):
        global_axes[i] = translation + rotation @ point
    
    # 坐标轴颜色 (红色X轴, 绿色Y轴, 蓝色Z轴)
    axes_colors = np.array([
        [255, 0, 0],  # X轴起点
        [255, 0, 0],  # X轴终点
        [0, 255, 0],  # Y轴起点
        [0, 255, 0],  # Y轴终点
        [0, 0, 255],  # Z轴起点
        [0, 0, 255]   # Z轴终点
    ], dtype=np.uint8)
    
    # 添加坐标轴线
    rr.log(f"robot/{link_name}/axes", rr.LineStrips3D(
        strips=global_axes.reshape(3, 2, 3),
        colors=axes_colors.reshape(3, 2, 3)
    ))

print(f"已加载URDF模型: {urdf_file}")
print(f"模型有 {model.nq} 个关节自由度")
print(f"可视化了 {len(link_ids)} 个链接")
