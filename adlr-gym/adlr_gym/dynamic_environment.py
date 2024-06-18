import numpy as np  # 导入numpy库用于数组和矩阵操作
import random  # 导入random库用于生成随机数
from adlr_gym.visualization import MapGenerator  # 从visualization模块导入MapGenerator类，用于可视化地图
from adlr_gym.Astar import AStar  # 从Astar模块导入AStar类，用于实现A*算法
import pickle  # 导入pickle库用于序列化和反序列化对象
import numpy as np  # 导入numpy库用于数组和矩阵操作
import random  # 导入random库用于生成随机数
from adlr_gym.visualization import MapGenerator  # 从visualization模块导入MapGenerator类，用于可视化地图
from adlr_gym.Astar import AStar  # 从Astar模块导入AStar类，用于实现A*算法
import pickle  # 导入pickle库用于序列化和反序列化对象

# 生成带有动态障碍物的地图
# 生成带有动态障碍物的地图
class DynamicObstacles:
    def __init__(self, static_obstacles, dynamic_obstacle_density, algorithm):
        self.static_obstacles = static_obstacles  # 静态障碍物地图
        self.height, self.width = static_obstacles.shape  # 获取地图的高度和宽度
        self.num_dynamic_obstacles = int(dynamic_obstacle_density * self.height * self.width)  # 计算动态障碍物的数量
        self.algorithm = algorithm  # 传入的路径规划算法
        self.dynamic_obstacles = self.initialize_dynamic_obstacles(self.num_dynamic_obstacles)  # 初始化动态障碍物
        self.paths = self.calculate_paths()  # 计算每个动态障碍物的路径
        self.current_positions = [obs[0] for obs in self.dynamic_obstacles]  # 获取所有动态障碍物的当前位置
        self.directions = [1] * self.num_dynamic_obstacles  # 1: 沿路径正方向，-1: 沿路径负方向
        self.static_obstacles = static_obstacles  # 静态障碍物地图
        self.height, self.width = static_obstacles.shape  # 获取地图的高度和宽度
        self.num_dynamic_obstacles = int(dynamic_obstacle_density * self.height * self.width)  # 计算动态障碍物的数量
        self.algorithm = algorithm  # 传入的路径规划算法
        self.dynamic_obstacles = self.initialize_dynamic_obstacles(self.num_dynamic_obstacles)  # 初始化动态障碍物
        self.paths = self.calculate_paths()  # 计算每个动态障碍物的路径
        self.current_positions = [obs[0] for obs in self.dynamic_obstacles]  # 获取所有动态障碍物的当前位置
        self.directions = [1] * self.num_dynamic_obstacles  # 1: 沿路径正方向，-1: 沿路径负方向

    def initialize_dynamic_obstacles(self, num_dynamic_obstacles):
        dynamic_obstacles = []
        while len(dynamic_obstacles) < num_dynamic_obstacles:
            start = (np.random.randint(self.height), np.random.randint(self.width))  # 随机生成起点
            goal = (np.random.randint(self.height), np.random.randint(self.width))  # 随机生成终点
            distance = np.linalg.norm(np.array(start) - np.array(goal))  # 计算起点和终点之间的距离
            # 确保起点和终点都不是静态障碍物且两者不相同
            if (not self.static_obstacles[start] and not self.static_obstacles[goal] and start != goal and 0 <= distance <= 5): 
                dynamic_obstacles.append((start, goal))# 将起点和终点加入动态障碍物列表
        return dynamic_obstacles

    def reset(self):
        self.obstacles = self.initialize_dynamic_obstacles(self.num_dynamic_obstacles)

    def calculate_paths(self):
        paths = []
        for start, goal in self.dynamic_obstacles:
            astar = self.algorithm(self.height, self.width, start, goal, self.static_obstacles)  # 创建A*算法实例
            path = astar.a_star_search()  # 使用A*算法计算路径
            paths.append(path)  # 将路径加入路径列表
        return paths

    def update_positions(self):
        new_positions = [] # 初始化一个空列表，用于存储更新后的每个障碍物的新位置
        # 循环遍历每个障碍物的当前位置和路径
        for i, (current_position, path) in enumerate(zip(self.current_positions, self.paths)):
            if not path:  # 如果路径为空，保持当前位置不变
                new_positions.append(current_position)
                continue  # 跳过本次for循环的剩余部分

            next_index = path.index(current_position) + self.directions[i] # 计算下一个位置的索引
            if 0 <= next_index < len(path):
                next_position = path[next_index]  # 获取下一个位置
                if next_position not in new_positions:  # 确保下一个位置未被其他障碍物占用
                    new_positions.append(next_position)
                    self.current_positions[i] = next_position
                else:
                    if random.random() < 0.5:  # 90%概率保持当前位置不变
                        new_positions.append(current_position)
                    else:  # 10%概率反向移动                        
                        if path.index(current_position) == 0 or path.index(current_position) == len(path) - 1:
                            new_positions.append(current_position)
                        else:
                            self.directions[i] = -self.directions[i]
                            next_index = path.index(current_position) + self.directions[i]
                            self.current_positions[i] = path[next_index]
                            new_positions.append(path[next_index])
            else:
                self.directions[i] = -self.directions[i]  # 到达路径终点后反向移动
                next_index = path.index(current_position) + self.directions[i]
                self.current_positions[i] = path[next_index]
                new_positions.append(path[next_index])

        self.current_positions = new_positions  # 更新当前位置

    def get_positions(self):
        return self.current_positions  # 返回所有动态障碍物的当前位置

# # 加载静态地图和全局路径
# with open('maps_and_paths.pkl', 'rb') as f:
#     maps_and_paths = pickle.load(f)

# # # 示例：可视化带有动态障碍物的地图
# static_obstacles, global_path = maps_and_paths[0]
# start = (0, 0)
# goal = (99, 99)
# dynamic_obstacle_density = 0.05
# dynamic_obstacles = DynamicObstacles(static_obstacles, dynamic_obstacle_density, AStar)
# MapGenerator.visualize_with_dynamic_obstacles(static_obstacles, start, goal, global_path, dynamic_obstacles)