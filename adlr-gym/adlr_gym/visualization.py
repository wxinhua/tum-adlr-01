import pickle  # 用于加载和保存对象的库
import matplotlib.pyplot as plt  # 用于绘图的库
import matplotlib.animation as animation  # 用于创建动画的库
import numpy as np  # 用于科学计算的库

# load map and global path
#with open('maps_and_paths.pkl', 'rb') as f:
    #maps_and_paths = pickle.load(f)


class MapGenerator:
    def __init__(self, height, width, static_obstacles):
        self.height = height  # 地图高度
        self.width = width  # 地图宽度
        self.static_obstacles = static_obstacles  # 静态障碍物地图


    # 可视化静态地图
    def visualize(self, start, goal, path=None):
        fig, ax = plt.subplots(figsize=(10, 10))  # 创建一个绘图对象
        ax.set_xlim(0, self.width)  # 设置x轴范围
        ax.set_ylim(0, self.height)  # 设置y轴范围
        ax.set_xticks(np.arange(0, self.width, 10))  # 设置x轴刻度
        ax.set_yticks(np.arange(0, self.height, 10))  # 设置y轴刻度
        ax.grid(True)  # 显示网格

        # 绘制静态障碍物
        for i in range(self.height):
            for j in range(self.width):
                if self.static_obstacles[i, j]:
                    ax.add_patch(plt.Rectangle((j, self.height - i - 1), 1, 1, color='black'))  # 绘制黑色方块表示障碍物

        # 绘制起点和终点
        start_patch = plt.Circle((start[1] + 0.5, self.height - start[0] - 0.5), 0.4, color='blue')
        goal_patch = plt.Circle((goal[1] + 0.5, self.height - goal[0] - 0.5), 0.4, color='red')
        ax.add_patch(start_patch)
        ax.add_patch(goal_patch)

        # 如果有路径，绘制路径
        if path:
            for (x, y) in path:
                ax.add_patch(plt.Rectangle((y, self.height - x - 1), 1, 1, color='green', alpha=0.5))  # 绘制绿色方块表示路径

        plt.show()  # 显示图像



    # 可视化带有动态障碍物的地图
    @staticmethod
    def visualize_with_dynamic_obstacles(static_obstacles, start, goal, path, dynamic_obstacles, num_steps=50):
        fig, ax = plt.subplots(figsize=(10, 10))  # 创建一个绘图对象
        ax.set_xlim(0, static_obstacles.shape[1])  # 设置x轴范围
        ax.set_ylim(0, static_obstacles.shape[0])  # 设置y轴范围
        ax.set_xticks(np.arange(0, static_obstacles.shape[1], 10))  # 设置x轴刻度
        ax.set_yticks(np.arange(0, static_obstacles.shape[0], 10))  # 设置y轴刻度
        ax.grid(True)  # 显示网格

        # 绘制静态障碍物
        for i in range(static_obstacles.shape[0]):
            for j in range(static_obstacles.shape[1]):
                if static_obstacles[i, j]:
                    ax.add_patch(plt.Rectangle((j, static_obstacles.shape[0] - i - 1), 1, 1, color='black'))  # 绘制黑色方块表示障碍物

        # 绘制起点和终点
        start_patch = plt.Circle((start[1] + 0.5, static_obstacles.shape[0] - start[0] - 0.5), 0.4, color='blue')
        goal_patch = plt.Circle((goal[1] + 0.5, static_obstacles.shape[0] - goal[0] - 0.5), 0.4, color='red')
        ax.add_patch(start_patch)
        ax.add_patch(goal_patch)

        # 如果有路径，绘制路径
        if path:
            for (x, y) in path:
                ax.add_patch(plt.Rectangle((y, static_obstacles.shape[0] - x - 1), 1, 1, color='green', alpha=0.5))  # 绘制绿色方块表示路径

        # 绘制每个动态障碍物的路径
        path_patches = []
        for path in dynamic_obstacles.paths:
            if path:
                for (x, y) in path:
                    path_patch = plt.Rectangle((y, static_obstacles.shape[0] - x - 1), 1, 1, color='lightblue', alpha=0.5)
                    path_patches.append(path_patch)
                    ax.add_patch(path_patch)


        # 绘制动态障碍物的初始位置
        dynamic_patches = []
        for (x, y) in dynamic_obstacles.get_positions():
            patch = plt.Rectangle((y, static_obstacles.shape[0] - x - 1), 1, 1, color='yellow', alpha=0.7)
            dynamic_patches.append(patch)
            ax.add_patch(patch)

        # 更新函数，用于动画
        def update(t):
            dynamic_obstacles.update_positions()  # 更新动态障碍物位置
            for patch in dynamic_patches:
                patch.remove()  # 移除旧位置的动态障碍物
            dynamic_patches.clear()  # 清空动态障碍物列表

            # 绘制动态障碍物的新位置
            for (x, y) in dynamic_obstacles.get_positions():
                patch = plt.Rectangle((y, static_obstacles.shape[0] - x - 1), 1, 1, color='yellow')
                dynamic_patches.append(patch)
                ax.add_patch(patch)

        # 创建动画
        ani = animation.FuncAnimation(fig, update, frames=num_steps, repeat=True)
        plt.show()  # 显示动画

# 检查第 Ith 张地图和全局路径
# static_obstacles, global_path = maps_and_paths[70]
# map_gen = MapGenerator(100, 100, static_obstacles)
# map_gen.visualize((0, 0), (99, 99), global_path)

# 可视化所有地图和全局路径
# for i, (static_obstacles, global_path) in enumerate(maps_and_paths):
#     print(f"Visualizing map {i+1}")
#     map_gen = MapGenerator(100, 100, static_obstacles)
#     map_gen.visualize((0, 0), (99, 99), global_path)
