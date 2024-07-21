import gymnasium as gym
import numpy as np
from gymnasium import spaces
import pygame
import matplotlib.pyplot as plt
from ..Astar import AStar  # A* algorithm for global path planning
from ..dynamic_environment import DynamicObstacles  # Manage dynamic obstacles
from ..static_environment import MapGenerator  # Manage static obstacles
from gymnasium.utils import seeding

class MapEnv(gym.Env):
    metadata = {'render_modes': ['human', 'rgb_array', 'None'],
                'render_fps': 10}
    def __init__(self, height=50, width=50, obstacle_density=0.2, dynamic_density=0.01, fov_size = 15, temporal_length=4, render_mode='human'):
        super(MapEnv, self).__init__()
        
        # Initialize static environment
        self.height = height
        self.width = width
        self.fov_size = fov_size
        self.obstacle_density = obstacle_density
        self.dynamic_density = dynamic_density
        self.current_episode = 0
        self.current_step = 0
        self.path_removed = 0
        self.last_path_index = 0
        self.temporal_length = temporal_length
        self.render_mode = render_mode
        self.on_path = True
        self.last_path_time = 0  # last time step on the global path
        self.time_since_off_path = 0  # temporal length after leaving global path
        self.last_off_path_index = 0
        
        # Define action and observation space
        self.action_space = spaces.Discrete(5)  # up, down, left, right, idle
        #self.observation_space = spaces.Box(low=0, high=255, shape=(self.temporal_length, fov_size, fov_size, 4), dtype=np.uint8)
        self.observation_space = spaces.Box(low=-1, high=2, shape=(self.fov_size, self.fov_size, 2), dtype=np.int16) #float
        #self.observations = np.zeros((self.temporal_length, self.fov_size, self.fov_size, 4), dtype=np.uint8)
        #self.observations = -np.ones((self.fov_size, self.fov_size, 2), dtype=np.uint8)   
        # Initialize pygame
        pygame.init()
        pygame.display.init()
        self.window_size = 1000 
        self.window = pygame.display.set_mode((self.window_size, self.window_size))
        self.clock = pygame.time.Clock()

    def seed(self, seed=None):
    
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def generate_static_obstacles(self):
        map_generator = MapGenerator(self.height, self.width, self.obstacle_density)
        return map_generator.generate_obstacles()
    
    def make_start_goal_clear(self):
        map_generator = MapGenerator(self.height, self.width, self.obstacle_density)
        return map_generator.ensure_start_goal_clear(self.start, self.goal)
    
    def calculate_global_path(self, start, goal, static_obstacles):
        astar = AStar(self.height, self.width, start, goal, static_obstacles)
        return astar.a_star_search()
    
    def generate_dynamic_obstacles(self, static_obstacles, dynamic_density):
        return DynamicObstacles(static_obstacles, dynamic_density, AStar)
    
    def select_random_positions(self):
        free_positions = np.argwhere(self.static_obstacles == 0)
        idx_start, idx_goal = np.random.choice(len(free_positions), 2, replace=False)
        return tuple(free_positions[idx_start]), tuple(free_positions[idx_goal])
    
    def reset(self, **kwargs):
        # Reset the state of the environment to an initial state
        
        self.static_obstacles = self.generate_static_obstacles()
        self.start, self.goal = self.select_random_positions()
        self.make_start_goal_clear()
        self.global_path = self.calculate_global_path(self.start, self.goal, self.static_obstacles)
        while not self.global_path:
            self.static_obstacles = self.generate_static_obstacles()
            self.start, self.goal = self.select_random_positions()
            self.make_start_goal_clear()
            self.global_path = self.calculate_global_path(self.start, self.goal, self.static_obstacles)

        self.dynamic_obstacles = self.generate_dynamic_obstacles(self.static_obstacles, self.dynamic_density)
        
        self.current_position = self.start
        self.current_step = 0
        self.current_episode += 1
        self.last_path_index = 0
        self.on_path = True
        self.last_path_time = 0  
        self.time_since_off_path = 0  
        self.last_off_path_index = 0
        if 'seed' in kwargs:
            self.seed(kwargs['seed'])

        # self.observations = np.zeros((self.temporal_length, self.fov_size, self.fov_size, 4), dtype=np.uint8)
        # for i in range(self.temporal_length):
        #     self.observations[i] = self._get_observation()
        self.observations = self._get_observation()

        info = {} 

        return self.observations, info

    def step(self, action):
        # Execute one time step within the environment
        self.current_step += 1
        next_position = self._move_robot(action)
        #print(f"Current Position: {self.current_position}, Action Taken: {action}, Next Position: {next_position}")
        self.current_position = next_position
        path_time = self.path_time_step(action)
         
        
        reward = self._calculate_reward(path_time=path_time, next_position=next_position)
        self.dynamic_obstacles.update_positions()

        # self.observations = np.roll(self.observations, -1, axis=0)
        # self.observations[-1] = self._get_observation()
        self.observations = self._get_observation()

        terminated = self._check_terminated(next_position)
        truncated = self._check_truncated()
        info = {}
        
        return self.observations, reward, terminated, truncated, info
    

    def path_time_step(self, action):
        path_time = 0
        if self.current_position in self.global_path:
            current_index = self.global_path.index(self.current_position)

            # agent returns back or always stays on the global path
            if not self.on_path:
                # agent returns back to global path
                self.on_path = True
                self.last_path_index = current_index  # update position when agent returns back to global path
                path_time = self.current_step - self.time_since_off_path
                self.time_since_off_path = 0

                # remove guidance between leaving and getting back
                if self.last_off_path_index != -1 and self.last_off_path_index < current_index:
                    del self.global_path[self.last_off_path_index:current_index]
        
            # the reward of each cell on the global guidance path can only be collected once 
            if action == 4:  # Idle
                del self.global_path[current_index]  
                self.last_path_index = current_index - 1 if current_index > 0 else -1  

            else:
                # other actions
                if current_index > self.last_path_index:
                    del self.global_path[self.last_path_index:current_index]
                self.last_path_index = current_index

        else:
            if self.on_path:
                # agent moves away from global path
                if self.last_path_index != -1:
                    del self.global_path[self.last_path_index]  
                self.on_path = False
                self.last_off_path_ondex = self.last_path_index  
                self.time_since_off_path = self.current_step
                self.last_path_index = -1  

        return path_time



    def _check_terminated(self, next_position):
        if self.current_position == self.goal:
            return True
        elif not self._has_global_guidance():
            return True
        elif self.static_obstacles[self.current_position] == 1 or next_position in self.dynamic_obstacles.get_positions():
            return True
        else:
            return False
    
    def _check_truncated(self):
        max_steps = 50 + 10 * self.path_time_step(self)
      
        #max_steps = 100
        if self.current_step >= max_steps:
            return True
        
        else:
            return False
    
    def _has_global_guidance(self):
        half_fov_h = self.fov_size // 2
        top, left = max(0, self.current_position[0] - half_fov_h), max(0, self.current_position[1] - half_fov_h)
        bottom, right = min(self.height, self.current_position[0] + half_fov_h + 1), min(self.width, self.current_position[1] + half_fov_h + 1)
        
        for i in range(top, bottom):
            for j in range(left, right):
                if (i, j) in self.global_path:
                    return True
        
        return False


    
    def _move_robot(self, action):
        # Update robot position based on action
        x, y = self.current_position
        next_position = (x, y)
        # Calculate potential new position based on the action
        if action == 0 :  # Up
            next_position = (max(x - 1, 0), y)
        elif action == 1 :  # Down
            next_position = (min(x + 1, self.height-1), y)
        elif action == 2 :  # Left
            next_position = (x, max(y - 1, 0))
        elif action == 3 :  # Right
            next_position = (x, min(y + 1, self.width-1))
        elif action == 4:  # Idle
            next_position = (x, y)
    
        return next_position


    def _calculate_reward(self, path_time, next_position):
        # Calculate reward
        r1, r2, r3 = -0.01, -0.1, 0.1
        # punishment for collision
        if self.static_obstacles[next_position] == 1 or next_position in self.dynamic_obstacles.get_positions():
            reward = -100

        # if next_position == self.current_position:
        #     reward = -2
        # large reward for reaching the goal
        elif next_position == self.goal:
            reward = 100
            
        elif next_position in self.global_path:
            if path_time > 0:  # reward for getting back to global path
                reward = -1 + path_time * 4
            else:
                reward = 10  # always following the global path
        else:
            reward = -1   # punishment for leaving

        return reward
     

    def _get_observation(self):
        
        # rgb_observation = self._get_local_observation()  # (fov_size, fov_size, 3)
        # guidance = self._global_guidance()               # (fov_size, fov_size)
        # guidance = np.expand_dims(guidance, axis=2)       # (fov_size, fov_size, 1)
        # observation = np.concatenate((rgb_observation, guidance), axis=2)  # (fov_size, fov_size, 4)
        observation = self._get_local_observation()
        guidance = self._global_guidance()
        observation = np.dstack((observation, guidance)) # (fov_size, fov_size, 2)
        
        return observation
    
    def _get_local_observation(self):
        # local_obs = np.zeros((self.fov_size, self.fov_size, 3), dtype=np.uint8)

        # # Initialize the observation grid with -1 (indicating free space)
        local_obs = -np.ones((self.fov_size, self.fov_size), dtype=np.int8)

        half_fov_h = self.fov_size // 2
        top, left = max(0, self.current_position[0] - half_fov_h), max(0, self.current_position[1] - half_fov_h)
        bottom, right = min(self.height, self.current_position[0] + half_fov_h + 1), min(self.width, self.current_position[1] + half_fov_h + 1)

        # for i in range(top, bottom):
        #     for j in range(left, right):
        #         local_row, local_col = i - top, j - left
                
        #         if self.static_obstacles[i, j] == 0:
        #             local_obs[local_row, local_col] = [255, 255, 255]  # white for free cells
        #         elif self.static_obstacles[i, j] == 1:
        #             local_obs[local_row, local_col] = [0, 255, 0]  # green for static obstacles
                
        #         if (i, j) in self.dynamic_obstacles.get_positions():
        #             local_obs[local_row, local_col] = [255, 255, 0]  # yellow for dynamic obstacles

        #         if (i, j) == self.current_position:
        #             local_obs[local_row, local_col]  = [0, 0, 255] # blue for agent

        #         if (i, j) in self.global_path:
        #             local_obs[local_row, local_col]  = [0, 0, 0]

        #Populate the local observation grid
        for i in range(top, bottom):
            for j in range(left, right):
                local_row, local_col = i - top, j - left
                if self.static_obstacles[i, j] == 1:
                    local_obs[local_row, local_col] = 1  # Static obstacles
                if (i, j) in self.dynamic_obstacles.get_positions():
                    local_obs[local_row, local_col] = 2  # Dynamic obstacles
                # if (i, j) in self.global_path:
                #     local_obs[local_row, local_col] = 0  # black for global path

        return local_obs
    
    def _global_guidance(self):
        guidance = -np.ones((self.fov_size, self.fov_size), dtype=np.uint8)
        half_fov_h = self.fov_size // 2
        top, left = max(0, self.current_position[0] - half_fov_h), max(0, self.current_position[1] - half_fov_h)
        bottom, right = min(self.height, self.current_position[0] + half_fov_h + 1), min(self.width, self.current_position[1] + half_fov_h + 1)
        
        for i in range(top, bottom):
            for j in range(left, right):
                local_row, local_col = i - top, j - left

                if (i, j) in self.global_path:
                    guidance[local_row, local_col] = 0  # black for global path

        return guidance


    def render(self):
        if self.metadata['render_modes'] == 'rgb_array':
            return self._render_frame()
        else:
            self._render_frame()

    def _render_frame(self):
        
        pix_square_size = self.window_size / max(self.width, self.height)

        canvas = pygame.Surface((self.window_size, self.window_size))
        canvas.fill((255, 255, 255))  

        
        for x in range(self.height):
            for y in range(self.width):
                rect = pygame.Rect(y * pix_square_size, x * pix_square_size, pix_square_size, pix_square_size)
                
                if self.static_obstacles[x, y] == 1:
                    pygame.draw.rect(canvas, (0, 255, 0), rect)  
                if (x, y) in self.global_path:
                    pygame.draw.rect(canvas, (0, 0, 0), rect)  
                if (x, y) in self.dynamic_obstacles.get_positions():
                    pygame.draw.rect(canvas, (255, 255, 0), rect)  

        
        agent_x, agent_y = self.current_position
        agent_rect = pygame.Rect(agent_y * pix_square_size, agent_x * pix_square_size, pix_square_size, pix_square_size)
        pygame.draw.ellipse(canvas, (0, 0, 255), agent_rect)  

        goal_x, goal_y = self.goal
        goal_rect = pygame.Rect(goal_y * pix_square_size, goal_x * pix_square_size, pix_square_size, pix_square_size)
        pygame.draw.ellipse(canvas, (255, 0, 0), goal_rect)

        self.window.blit(canvas, (0, 0))
        pygame.display.update()
        self.clock.tick(self.metadata['render_fps'])  # fps

  

    def close(self):
        pygame.quit()
