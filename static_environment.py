import numpy as np  # 导入numpy库，用于数值计算
import matplotlib.pyplot as plt  # 导入matplotlib库中的pyplot模块，用于绘图

# 生成包含静态障碍物的地图
class MapGenerator:
    def __init__(self, height, width, obstacles_density):
        # 初始化地图生成器
        self.height = height  # 地图高度
        self.width = width  # 地图宽度
        self.obstacles_density = obstacles_density  # 障碍物密度
        self.static_obstacles = self.generate_obstacles()  # 生成静态障碍物

    def generate_obstacles(self):
        # 生成静态障碍物
        return np.random.rand(self.height, self.width) < self.obstacles_density  # 随机生成障碍物矩阵

    def ensure_start_goal_clear(self, start, goal):
        # 确保起始点和目标点没有障碍物
        self.static_obstacles[start] = False  # 设置起始点为无障碍
        self.static_obstacles[goal] = False  # 设置目标点为无障碍

    def visualize(self, start, goal, path=None):
        # 可视化地图
        fig, ax = plt.subplots(figsize=(10, 10))  # 创建一个绘图窗口
        ax.set_xlim(0, self.width)  # 设置x轴范围
        ax.set_ylim(0, self.height)  # 设置y轴范围
        ax.set_xticks(np.arange(0, self.width, 10))  # 设置x轴刻度
        ax.set_yticks(np.arange(0, self.height, 10))  # 设置y轴刻度
        ax.grid(True)  # 显示网格

        for i in range(self.height):
            for j in range(self.width):
                if self.static_obstacles[i, j]:
                    ax.add_patch(plt.Rectangle((j, self.height - i - 1), 1, 1, color='black'))  # 绘制障碍物

        # 绘制起始点和目标点
        start_patch = plt.Circle((start[1] + 0.5, self.height - start[0] - 0.5), 0.4, color='blue')  # 绘制起始点
        goal_patch = plt.Circle((goal[1] + 0.5, self.height - goal[0] - 0.5), 0.4, color='red')  # 绘制目标点
        ax.add_patch(start_patch)
        ax.add_patch(goal_patch)

        if path:
            # 如果有路径，绘制路径
            for (x, y) in path:
                ax.add_patch(plt.Rectangle((y, self.height - x - 1), 1, 1, color='green', alpha=0.5))  # 绘制路径

        plt.show()  # 显示绘图




