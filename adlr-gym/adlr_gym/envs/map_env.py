import gymnasium as gym  # 导入gymnasium库
import numpy as np  # 导入numpy库
from gymnasium import spaces  # 从gymnasium库导入spaces模块
import pygame  # 导入pygame库
import matplotlib.pyplot as plt  # 导入matplotlib库中的pyplot模块
from adlr_gym.Astar import AStar  # 导入A*算法用于全局路径规划
from adlr_gym.dynamic_environment import DynamicObstacles  # 导入管理动态障碍物的模块
from adlr_gym.static_environment import MapGenerator  # 导入管理静态障碍物的模块
from gymnasium.utils import seeding  # 从gymnasium库导入seeding模块

class MapEnv(gym.Env):
    metadata = {'render_modes': ['human', 'rgb_array', 'None'],
                'render_fps': 1}

    def __init__(self, height=50, width=50, obstacle_density=0.2, dynamic_density=0.05, fov_size=15, temporal_length=4, render_mode='human'):
        super(MapEnv, self).__init__()
        
        # 初始化静态环境
        self.height = height
        self.width = width
        self.fov_size = fov_size
        self.obstacle_density = obstacle_density
        self.dynamic_density = dynamic_density
        self.current_episode = 0
        self.current_step = 0
        self.path_removed = 0
        self.last_path_index = -1
        self.temporal_length = temporal_length
        self.render_mode = render_mode
        
        # 定义动作和观察空间
        self.action_space = spaces.Discrete(5)  # 定义动作空间为5个离散动作：上、下、左、右、静止
        self.observation_space = spaces.Box(low=0, high=255, shape=(self.temporal_length, fov_size, fov_size, 4), dtype=np.uint8)
        self.observations = np.zeros((self.temporal_length, self.fov_size, self.fov_size, 4), dtype=np.uint8)
        
        # 初始化pygame
        pygame.init()
        pygame.display.init()
        self.window_size = 1000 
        self.window = pygame.display.set_mode((self.window_size, self.window_size))
        self.clock = pygame.time.Clock()

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def generate_static_obstacles(self):
        # 生成静态障碍物
        map_generator = MapGenerator(self.height, self.width, self.obstacle_density)
        return map_generator.generate_obstacles()

    def calculate_global_path(self, start, goal, static_obstacles):
        # 计算全局路径
        astar = AStar(self.height, self.width, start, goal, static_obstacles)
        return astar.a_star_search()

    def generate_dynamic_obstacles(self, static_obstacles, dynamic_density):
        # 生成动态障碍物
        return DynamicObstacles(static_obstacles, dynamic_density, AStar)

    def select_random_positions(self):
        # 随机选择起始和目标位置
        free_positions = np.argwhere(self.static_obstacles == 0)
        idx_start, idx_goal = np.random.choice(len(free_positions), 2, replace=False)
        return tuple(free_positions[idx_start]), tuple(free_positions[idx_goal])

    def reset(self, **kwargs):
        # 重置环境状态
        self.static_obstacles = self.generate_static_obstacles()
        self.start, self.goal = self.select_random_positions()
        self.global_path = self.calculate_global_path(self.start, self.goal, self.static_obstacles)
        while not self.global_path:
            self.static_obstacles = self.generate_static_obstacles()
            self.start, self.goal = self.select_random_positions()
            self.global_path = self.calculate_global_path(self.start, self.goal, self.static_obstacles)

        self.dynamic_obstacles = self.generate_dynamic_obstacles(self.static_obstacles, self.dynamic_density)
        self.current_position = self.start
        self.current_step = 0
        self.current_episode += 1
        self.path_removed = 0
        self.last_path_index = -1

        if 'seed' in kwargs:
            self.seed(kwargs['seed'])

        self.observations = np.zeros((self.temporal_length, self.fov_size, self.fov_size, 4), dtype=np.uint8)
        for i in range(self.temporal_length):
            self.observations[i] = self._get_observation()

        info = {} 
        return self.observations, info

    def step(self, action):
        # 执行一个时间步长的动作
        self.current_step += 1
        next_position = self._move_robot(action)
        
        # 更新全局路径索引
        if next_position in self.global_path:
            current_index = self.global_path.index(next_position)
            # 检查是否为连续路径或重新进入路径
            if self.last_path_index == -1 or abs(current_index - self.last_path_index) == 1:
                self.last_path_index = current_index
                self.global_path = self.global_path[current_index:]
            else:
                # 如果不连续
                self.last_path_index = -1
        else:
            # 如果移出全局路径，根据策略可能需要重置索引
            self.last_path_index = -1  # 重置或根据需要调整

        reward = self._calculate_reward(self.current_position, next_position) # 计算从当前位置移动到下一个位置的奖励
        self._update_path_tracking() # 更新路径跟踪信息
        self.dynamic_obstacles.update_positions()  # 更新动态障碍物的位置
        self.observations = np.roll(self.observations, -1, axis=0) # 将观察值矩阵沿着时间轴滚动一位，使最新的观察值在最后一位
        self.observations[-1] = self._get_observation() # 获取新的观察值并存储在观察值矩阵的最后一位
        
        terminated = self._check_terminated()
        truncated = self._check_truncated()
        info = {}
        return self.observations, reward, terminated, truncated, info

    def _check_terminated(self):
        # 检查任务是否完成
        if self.current_position == self.goal:
            return True
        return False

    def _check_truncated(self):
        # 检查任务是否被截断
        max_steps = 50 + 10 * self.path_removed
        if self.current_step >= max_steps or not self._has_global_guidance():
            return True
        else:return False

    def _has_global_guidance(self):
        # 检查机器人是否在全局引导路径上
        half_fov_h = self.fov_size // 2
        top, left = max(0, self.current_position[0] - half_fov_h), max(0, self.current_position[1] - half_fov_h)
        bottom, right = min(self.height, self.current_position[0] + half_fov_h + 1), min(self.width, self.current_position[1] + half_fov_h + 1)
        
        for i in range(top, bottom):
            for j in range(left, right):
                if (i, j) in self.global_path:
                    return True
        return False

    def _move_robot(self, action):
        # 更新机器人位置
        x, y = self.current_position
        next_position = (x, y)
        if action == 0 and x > 0:  # 向上移动
            next_position = (x - 1, y)
        elif action == 1 and x < self.height - 1:  # 向下移动
            next_position = (x + 1, y)
        elif action == 2 and y > 0:  # 向左移动
            next_position = (x, y - 1)
        elif action == 3 and y < self.width - 1:  # 向右移动
            next_position = (x, y + 1)
        elif action == 4:  # 静止
            next_position = (x, y)

        # 检查下一个位置是否没有障碍物
        if self.static_obstacles[next_position] == 0:
            self.current_position = next_position  
        return self.current_position

    def _calculate_reward(self, current_position, next_position):
        # 计算奖励
        # 定义四种不同的奖励值
        r1, r2, r3, r4 = -0.5, -1, 3, -5
        
        # 初始化last_position 和 loop_count
        if not hasattr(self, 'last_position'):
            self.last_position = None
        if not hasattr(self, 'loop_count'):
            self.loop_count = 0
        self.bounce_threshold = 3

        # 如果下一个位置是静态障碍物或动态障碍物
        if self.static_obstacles[next_position] == 1 or next_position in self.dynamic_obstacles.get_positions():
            return r1 + r2  # 遇到障碍，返回较大的负奖励

        # 如果agent不动
        if current_position == next_position:
            return r2  # 返回一个较大的负奖励
        
        # 检查机器人是否在全局引导路径上
        if next_position in self.global_path:
            # 如果是第一次进入全局路径
            if self.last_path_index == -1:
                self.last_path_index = self.global_path.index(next_position)
                return 0  # 第一次进入路径，不计算额外奖励
            else:
                current_index = self.global_path.index(next_position)
                # 如果当前位置在上次路径索引之后（即机器人在向目标前进）
                if current_index > self.last_path_index:
                    path_removed = current_index - self.last_path_index - 1  # 计算离开路径的步数
                    self.last_path_index = current_index  # 更新路径索引
                    return r1 + path_removed * r3  # 奖励为基础奖励加上路径奖励
                else:
                    return r1  # 如果没有前进，返回基础奖励
        else:
            # 如果不在全局路径上，返回基础奖励
            reward = r1
            
            # 检测agent是否在两个特定点之间来回跳动
            if self.last_position == next_position:
                self.loop_count += 1
                if self.loop_count >= self.bounce_threshold:
                    reward += r2  # 如果达到阈值，给予较大的负奖励
            else:
                self.loop_count = 0  # 重置循环计数器

            self.last_position = current_position  # 更新上一个位置
            return reward


    # def _calculate_reward(self, current_position, next_position):
    #     # Calculate reward
    #     r1, r2, r3 = -0.01, -0.1, 0.1
    
    #     if self.static_obstacles[next_position] == 1 or next_position in self.dynamic_obstacles.get_positions():
    #         return r1 + r2 
    #     if current_position == next_position:
    #         return r2
    #     if next_position in self.global_path:
    #         # 如果是第一次进入全局路径
    #         if self.last_path_index == -1:
    #             self.last_path_index = self.global_path.index(next_position)
    #             return 0  # 第一次进入路径，不计算额外奖励
    #         else:
    #             current_index = self.global_path.index(next_position)
    #             # 如果当前位置在上次路径索引之后（即机器人在向目标前进）
    #             if current_index > self.last_path_index:
    #                 path_removed = current_index - self.last_path_index - 1  # 计算离开路径的步数
    #                 self.last_path_index = current_index  # 更新路径索引
    #                 return r1 + path_removed * r3  # 奖励为基础奖励加上路径奖励
    #             else:
    #                 return r1  # 如果没有前进，返回基础奖励
    #     else:
    #         return r1 


    def _update_path_tracking(self):
        current_index = 0  # 给 current_index 一个默认值
        if self.current_position in self.global_path:
            current_index = self.global_path.index(self.current_position)
        if self.last_path_index != -1:
            self.path_removed = abs(current_index - self.last_path_index) - 1
            self.last_path_index = current_index
        else:
            self.last_path_index = -1
            self.path_removed += 1
        return self.path_removed

    def _get_observation(self):
        # 获取观察值
        rgb_observation = self._get_local_observation()  # 获取局部观察值
        guidance = self._global_guidance()  # 获取全局引导信息
        guidance = np.expand_dims(guidance, axis=2)  # 扩展维度
        observation = np.concatenate((rgb_observation, guidance), axis=2)  # 拼接观察值和引导信息
        return observation

    def _get_local_observation(self):
        # 获取局部观察值
        local_obs = np.zeros((self.fov_size, self.fov_size, 3), dtype=np.uint8)
        half_fov_h = self.fov_size // 2
        top, left = max(0, self.current_position[0] - half_fov_h), max(0, self.current_position[1] - half_fov_h)
        bottom, right = min(self.height, self.current_position[0] + half_fov_h + 1), min(self.width, self.current_position[1] + half_fov_h + 1)

        for i in range(top, bottom):
            for j in range(left, right):
                local_row, local_col = i - top, j - left
                
                if self.static_obstacles[i, j] == 0:
                    local_obs[local_row, local_col] = [255, 255, 255]  # 空闲格子为白色
                elif self.static_obstacles[i, j] == 1:
                    local_obs[local_row, local_col] = [0, 255, 0]  # 静态障碍物为绿色
                
                if (i, j) in self.dynamic_obstacles.get_positions():
                    local_obs[local_row, local_col] = [255, 255, 0]  # 动态障碍物为黄色

                if (i, j) == self.current_position:
                    local_obs[local_row, local_col] = [0, 0, 255]  # 机器人为蓝色

        return local_obs

    def _global_guidance(self):
        # 获取全局引导信息
        guidance = np.zeros((self.fov_size, self.fov_size), dtype=np.uint8)
        half_fov_h = self.fov_size // 2
        top, left = max(0, self.current_position[0] - half_fov_h), max(0, self.current_position[1] - half_fov_h)
        bottom, right = min(self.height, self.current_position[0] + half_fov_h + 1), min(self.width, self.current_position[1] + half_fov_h + 1)
        
        for i in range(top, bottom):
            for j in range(left, right):
                local_row, local_col = i - top, j - left

                if (i, j) in self.global_path:
                    guidance[local_row, local_col] = 0  # 全局路径为黑色

        return guidance

    def render(self):
        # 渲染环境
        if self.metadata['render_modes'] == 'rgb_array':
            return self._render_frame()
        else:
            self._render_frame()

    def _render_frame(self):
        # 渲染帧
        pix_square_size = self.window_size / max(self.width, self.height)
        canvas = pygame.Surface((self.window_size, self.window_size))
        canvas.fill((255, 255, 255))  

        for x in range(self.height):
            for y in range(self.width):
                rect = pygame.Rect(y * pix_square_size, x * pix_square_size, pix_square_size, pix_square_size)
                
                if self.static_obstacles[x, y] == 1:
                    pygame.draw.rect(canvas, (0, 255, 0), rect)  # 静态障碍物为绿色
                if (x, y) in self.global_path:
                    pygame.draw.rect(canvas, (0, 0, 0), rect)  # 全局路径为黑色
                if (x, y) in self.dynamic_obstacles.get_positions():
                    pygame.draw.rect(canvas, (255, 255, 0), rect)  # 动态障碍物为黄色

        agent_x, agent_y = self.current_position
        agent_rect = pygame.Rect(agent_y * pix_square_size, agent_x * pix_square_size, pix_square_size, pix_square_size)
        pygame.draw.ellipse(canvas, (0, 0, 255), agent_rect)  # 机器人为蓝色

        goal_x, goal_y = self.goal
        goal_rect = pygame.Rect(goal_y * pix_square_size, goal_x * pix_square_size, pix_square_size, pix_square_size)
        pygame.draw.ellipse(canvas, (255, 0, 0), goal_rect)  # 目标为红色

        self.window.blit(canvas, (0, 0))
        pygame.display.update()
        # self.clock.tick(self.metadata['render_fps'])  # fps
        self.clock.tick(10) 
  

    def close(self):
        # 关闭环境
        pygame.quit()



# import gymnasium as gym
# import numpy as np
# from gymnasium import spaces
# import pygame
# import matplotlib.pyplot as plt
# from ..Astar import AStar  # A* algorithm for global path planning
# from ..dynamic_environment import DynamicObstacles  # Manage dynamic obstacles
# from ..static_environment import MapGenerator  # Manage static obstacles
# from gymnasium.utils import seeding

# class MapEnv(gym.Env):
#     metadata = {'render_modes': ['human', 'rgb_array', 'None'],
#                 'render_fps': 1}
#     def __init__(self, height=50, width=50, obstacle_density=0.2, dynamic_density=0.0, fov_size = 15, temporal_length=4, render_mode='human'):
#         super(MapEnv, self).__init__()
        
#         # Initialize static environment
#         self.height = height
#         self.width = width
#         self.fov_size = fov_size
#         self.obstacle_density = obstacle_density
#         self.dynamic_density = dynamic_density
#         #self.map_generator = MapGenerator(height, width, self.obstacle_density)
#         #self.static_obstacles = self.map_generator.generate_obstacles()
#         self.current_episode = 0
#         self.current_step = 0
#         self.path_removed = 0
#         self.last_path_index = -1
#         self.temporal_length = temporal_length
#         self.render_mode = render_mode
#         # Initialize the start and goal positions
#         #self.start = (0, 0)
#         #self.goal = (height - 1, width - 1)
#         #self.map_generator.ensure_start_goal_clear(self.start, self.goal)
#         #self.current_position = self.start
        
#         # Initialize A* algorithm for global path planning
#         #self.astar = AStar(height, width, self.start, self.goal, self.static_obstacles)
#         #self.global_path = self.astar.a_star_search()
#         # Initialize dynamic environment
#         #self.dynamic_obstacles = DynamicObstacles(self.static_obstacles, dynamic_density, AStar)

#         # Define action and observation space
#         self.action_space = spaces.Discrete(5)  # up, down, left, right, idle
#         self.observation_space = spaces.Box(low=0, high=255, shape=(self.temporal_length, fov_size, fov_size, 4), dtype=np.uint8)
#         self.observations = np.zeros((self.temporal_length, self.fov_size, self.fov_size, 4), dtype=np.uint8)  
#         # Initialize pygame
#         pygame.init()
#         pygame.display.init()
#         self.window_size = 1000 
#         self.window = pygame.display.set_mode((self.window_size, self.window_size))
#         self.clock = pygame.time.Clock()

#     def seed(self, seed=None):
    
#         self.np_random, seed = seeding.np_random(seed)
#         return [seed]

#     def generate_static_obstacles(self):
#         map_generator = MapGenerator(self.height, self.width, self.obstacle_density)
#         return map_generator.generate_obstacles()
    
#     def calculate_global_path(self, start, goal, static_obstacles):
#         astar = AStar(self.height, self.width, start, goal, static_obstacles)
#         return astar.a_star_search()
    
#     def generate_dynamic_obstacles(self, static_obstacles, dynamic_density):
#         return DynamicObstacles(static_obstacles, dynamic_density, AStar)
    
#     def select_random_positions(self):
#         free_positions = np.argwhere(self.static_obstacles == 0)
#         idx_start, idx_goal = np.random.choice(len(free_positions), 2, replace=False)
#         return tuple(free_positions[idx_start]), tuple(free_positions[idx_goal])
    
#     def reset(self, **kwargs):
#         # Reset the state of the environment to an initial state
        
#         self.static_obstacles = self.generate_static_obstacles()
#         self.start, self.goal = self.select_random_positions()
#         self.global_path = self.calculate_global_path(self.start, self.goal, self.static_obstacles)
#         while not self.global_path:
#             self.static_obstacles = self.generate_static_obstacles()
#             self.start, self.goal = self.select_random_positions()
#             self.global_path = self.calculate_global_path(self.start, self.goal, self.static_obstacles)

#         self.dynamic_obstacles = self.generate_dynamic_obstacles(self.static_obstacles, self.dynamic_density)
        
#         self.current_position = self.start
#         self.current_step = 0
#         self.current_episode += 1
#         self.path_removed = 0
#         self.last_path_index = -1
#         if 'seed' in kwargs:
#             self.seed(kwargs['seed'])

#         self.observations = np.zeros((self.temporal_length, self.fov_size, self.fov_size, 4), dtype=np.uint8)
#         for i in range(self.temporal_length):
#             self.observations[i] = self._get_observation()

#         info = {} 

#         return self.observations, info

#     def step(self, action):
#         # Execute one time step within the environment
#         self.current_step += 1
#         next_position = self._move_robot(action)
        
#          # 更新全局路径索引
#         if next_position in self.global_path:
#             current_index = self.global_path.index(next_position)
#         # 检查是否为连续路径或重新进入路径
#             if self.last_path_index == -1 or abs(current_index - self.last_path_index) == 1:
#                 self.last_path_index = current_index
#             else:
#                 # 如果不连续，可能需要重置或进行特残处理
#                 self.last_path_index = -1
#         else:
#         # 如果移出全局路径，根据策略可能需要重置索引
#             self.last_path_index = -1  # 重置或根据需要调整
#         reward = self._calculate_reward(self.current_position, next_position)
#         self._update_path_tracking()
#         self.dynamic_obstacles.update_positions()
#         self.observations = np.roll(self.observations, -1, axis=0)
#         self.observations[-1] = self._get_observation()
        
#         terminated = self._check_terminated()
#         truncated = self._check_truncated()
#         info = {}
#         return self.observations, reward, terminated, truncated, info

#     def _check_terminated(self):
#         if self.current_position == self.goal:
#             return True
    
#         return False
    
#     def _check_truncated(self):
#         max_steps = 50 + 10 * self.path_removed
#         if self.current_step >= max_steps:
#             return True
#         if not self._has_global_guidance():
#             return True
        
#         return False
    
#     def _has_global_guidance(self):
#         half_fov_h = self.fov_size // 2
#         top, left = max(0, self.current_position[0] - half_fov_h), max(0, self.current_position[1] - half_fov_h)
#         bottom, right = min(self.height, self.current_position[0] + half_fov_h + 1), min(self.width, self.current_position[1] + half_fov_h + 1)
        
#         for i in range(top, bottom):
#             for j in range(left, right):
#                 if (i, j) in self.global_path:
#                     return False
        
#         return True


    
#     def _move_robot(self, action):
#         # Update robot position based on action
#         x, y = self.current_position
#         next_position = (x, y)
#         # Calculate potential new position based on the action
#         if action == 0 and x > 0:  # Up
#             next_position = (x - 1, y)
#         elif action == 1 and x < self.height - 1:  # Down
#             next_position = (x + 1, y)
#         elif action == 2 and y > 0:  # Left
#             next_position = (x, y - 1)
#         elif action == 3 and y < self.width - 1:  # Right
#             next_position = (x, y + 1)
#         elif action == 4:  # Idle
#             next_position = (x, y)

#         # Check if the next position is free from obstacles
#         if self.static_obstacles[next_position] == 0:  
#             self.current_position = next_position  
    
#         return self.current_position


#     def _calculate_reward(self, current_position, next_position):
#         # Calculate reward
#         r1, r2, r3 = -0.01, -0.1, 0.1
        
#         """ if next_position in self.global_path and current_position not in self.global_path:
#         # A large positive reward r 1 + N e × r 3 when the robot reaches one of the cells on the global guidance path, 
#         # where r 3 > |r 1 | > 0 and Ne is the number of cells removed from the global guidance path, 
#         # between the point where the robot first left that path, to the point where it rejoins it.
#             index_current = self.global_path.index(current_position)
#             index_next = self.global_path.index(next_position)
#             path_removed = abs(index_next - index_current) - 1
#             reward = r1 + (self._update_path_tracking() * r3)

#         if self.static_obstacles[next_position[0], next_position[1]] == 1 or next_position in self.dynamic_obstacles.get_positions():
#         # A large negative reward r 1 + r 2 when the robot conflicts with a static obstacle or a dynamic obstacle
#             reward = r1 + r2

#         elif next_position == self.goal:
#             # found the goal
#             reward = 1
#         else:
#              #A small negative reward r 1 < 0 when the robot reaches a free
#             # cell which is not located on the global guidance
#             reward = r1 """
#         """ if next_position == self.goal:
#             return 1 """
#         if self.static_obstacles[next_position] == 1 or next_position in self.dynamic_obstacles.get_positions():
#             return r1 + r2  # 遇到障碍，返回较大的负奖励
    
#         # 检查机器人是否在全局引导路径上
#         if next_position in self.global_path:
#         # 机器人在全局引导路径上
#             if self.last_path_index == -1:
#                 self.last_path_index = self.global_path.index(next_position)
#                 return 0  # 第一次进入路径，不计算额外奖励
#             else:
#                 current_index = self.global_path.index(next_position)
#                 if current_index > self.last_path_index:
#                     path_removed = current_index - self.last_path_index - 1
#                     self.last_path_index = current_index
#                     return r1 + path_removed * r3
#                 else:
#                     return r1
#         else:
#         # 机器人不在全局引导路径上，并且在自由格子上
#             return r1
    

    
#     def _update_path_tracking(self):
#         if self.current_position in self.global_path:
#             current_index = self.global_path.index(self.current_position)
#         if self.last_path_index != -1:  # agent on the global path
#             self.path_removed = abs(current_index - self.last_path_index) - 1
#             self.last_path_index = current_index
#         else:
#             self.last_path_index = -1  # not on
#             self.path_removed += 1  # accumulate the steps 
        
#         return self.path_removed


#     def _get_observation(self):
#         """ observation = np.zeros((self.height, self.width, 3), dtype=np.uint8)

#         # white for free cells and green for static obstacles
#         for i in range(self.height):
#             for j in range(self.width):
#                 if self.static_obstacles[i, j] == 0:
#                     observation[i, j] = [255, 255, 255]
#                 elif self.static_obstacles[i, j] == 1:
#                     observation[i, j] = [0, 255, 0]
        
#         # black for global path
#         for (i, j) in self.global_path:
#             observation[i, j] = [0, 0, 0]

#         # yellow for dynamic obstacles
#         for (i, j) in self.dynamic_obstacles.get_positions():
#             observation[i, j] = [255, 255, 0]

#         # blue for current position
#         observation[self.current_position[0], self.current_position[1]] = [0, 0, 255]
#         # red for goal position
#         observation[self.goal[0], self.goal[1]] = [255, 0, 0] 
#      """
#         rgb_observation = self._get_local_observation()  # (fov_size, fov_size, 3)
#         guidance = self._global_guidance()               # (fov_size, fov_size)
#         guidance = np.expand_dims(guidance, axis=2)       # (fov_size, fov_size, 1)
#         observation = np.concatenate((rgb_observation, guidance), axis=2)  # (fov_size, fov_size, 4)
        
#         return observation
    
#     def _get_local_observation(self):
#         local_obs = np.zeros((self.fov_size, self.fov_size, 3), dtype=np.uint8)
#         half_fov_h = self.fov_size // 2
#         top, left = max(0, self.current_position[0] - half_fov_h), max(0, self.current_position[1] - half_fov_h)
#         bottom, right = min(self.height, self.current_position[0] + half_fov_h + 1), min(self.width, self.current_position[1] + half_fov_h + 1)

#         for i in range(top, bottom):
#             for j in range(left, right):
#                 local_row, local_col = i - top, j - left
                
#                 if self.static_obstacles[i, j] == 0:
#                     local_obs[local_row, local_col] = [255, 255, 255]  # white for free cells
#                 elif self.static_obstacles[i, j] == 1:
#                     local_obs[local_row, local_col] = [0, 255, 0]  # green for static obstacles
                
#                 if (i, j) in self.dynamic_obstacles.get_positions():
#                     local_obs[local_row, local_col] = [255, 255, 0]  # yellow for dynamic obstacles

#                 if (i, j) == self.current_position:
#                     local_obs[local_row, local_col]  = [0, 0, 255] # blue for agent

#         return local_obs
    
#     def _global_guidance(self):
#         guidance = np.zeros((self.fov_size, self.fov_size), dtype=np.uint8)
#         half_fov_h = self.fov_size // 2
#         top, left = max(0, self.current_position[0] - half_fov_h), max(0, self.current_position[1] - half_fov_h)
#         bottom, right = min(self.height, self.current_position[0] + half_fov_h + 1), min(self.width, self.current_position[1] + half_fov_h + 1)
        
#         for i in range(top, bottom):
#             for j in range(left, right):
#                 local_row, local_col = i - top, j - left

#                 if (i, j) in self.global_path:
#                     guidance[local_row, local_col] = 0  # black for global path

#         return guidance


#     """    def render(self, mode='human'):
#         if mode == 'human':
#             plt.imshow(self._get_observation())
#             plt.axis('on')
#         elif mode == 'rgb_array':
#             return self._get_observation() """

    
#     def render(self):
#         if self.metadata['render_modes'] == 'rgb_array':
#             return self._render_frame()
#         else:
#             self._render_frame()

#     def _render_frame(self):
        
#         pix_square_size = self.window_size / max(self.width, self.height)

#         canvas = pygame.Surface((self.window_size, self.window_size))
#         canvas.fill((255, 255, 255))  

        
#         for x in range(self.height):
#             for y in range(self.width):
#                 rect = pygame.Rect(y * pix_square_size, x * pix_square_size, pix_square_size, pix_square_size)
                
#                 if self.static_obstacles[x, y] == 1:
#                     pygame.draw.rect(canvas, (0, 255, 0), rect)  
#                 if (x, y) in self.global_path:
#                     pygame.draw.rect(canvas, (0, 0, 0), rect)  
#                 if (x, y) in self.dynamic_obstacles.get_positions():
#                     pygame.draw.rect(canvas, (255, 255, 0), rect)  

        
#         agent_x, agent_y = self.current_position
#         agent_rect = pygame.Rect(agent_y * pix_square_size, agent_x * pix_square_size, pix_square_size, pix_square_size)
#         pygame.draw.ellipse(canvas, (0, 0, 255), agent_rect)  

#         goal_x, goal_y = self.goal
#         goal_rect = pygame.Rect(goal_y * pix_square_size, goal_x * pix_square_size, pix_square_size, pix_square_size)
#         pygame.draw.ellipse(canvas, (255, 0, 0), goal_rect)

#         self.window.blit(canvas, (0, 0))
#         pygame.display.update()
#         self.clock.tick(self.metadata['render_fps'])  # fps

  

#     def close(self):
#         pygame.quit()


