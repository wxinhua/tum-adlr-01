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

    def __init__(self, height=100, width=100, obstacle_density=0.2, dynamic_density=0, fov_size = 15, temporal_length=4, render_mode='human'):
        super(MapEnv, self).__init__()
        
        # Initialize static environment
        self.height = height
        self.width = width
        self.fov_size = fov_size
        self.obstacle_density = obstacle_density
        self.dynamic_density = dynamic_density
        self.map_generator = MapGenerator(height, width, self.obstacle_density)
        self.static_obstacles = self.map_generator.generate_obstacles()
        self.current_episode = 0
        self.current_step = 0
        self.path_removed = 0
        self.last_path_index = -1
        self.temporal_length = temporal_length
        self.render_mode = render_mode
        # Initialize the start and goal positions
        self.start = (0, 0)
        self.goal = (height - 1, width - 1)
        self.map_generator.ensure_start_goal_clear(self.start, self.goal)
        self.current_position = self.start
        
        # Initialize A* algorithm for global path planning
        self.astar = AStar(height, width, self.start, self.goal, self.static_obstacles)
        self.global_path = self.astar.a_star_search()
        # Initialize dynamic environment
        self.dynamic_obstacles = DynamicObstacles(self.static_obstacles, dynamic_density, AStar)

        # Define action and observation space
        self.action_space = spaces.Discrete(5)  # up, down, left, right, idle
        self.observation_space = spaces.Box(low=0, high=255, shape=(self.temporal_length, fov_size, fov_size, 4), dtype=np.uint8)
        self.observations = np.zeros((self.temporal_length, self.fov_size, self.fov_size, 4), dtype=np.uint8)  
        # Initialize pygame
        pygame.init()
        pygame.display.init()
        self.window_size = 1000 
        self.window = pygame.display.set_mode((self.window_size, self.window_size))
        self.clock = pygame.time.Clock()

    def seed(self, seed=None):
    
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def reset(self, **kwargs):
        # Reset the state of the environment to an initial state
        #self.static_obstacles = self.map_generator.generate_obstacles()
        #self.map_generator.ensure_start_goal_clear(self.start, self.goal)
        #self.global_path = self.astar.a_star_search()
        self.dynamic_obstacles.reset()
        self.current_position = (0,0)
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

        return self.observations

    def step(self, action):
        # Execute one time step within the environment
        self.current_step += 1
        next_position = self._move_robot(action)
        self._update_path_tracking()
        self.dynamic_obstacles.update_positions()
        self.observations = np.roll(self.observations, -1, axis=0)
        self.observations[-1] = self._get_observation()
        reward = self._calculate_reward(self.current_position, next_position)
        terminated = self._check_terminated()
        truncated = self._check_truncated()
        info = {}
        return self.observations, reward, terminated, truncated, info

    def _check_terminated(self):
        if self.current_position == self.goal:
            return True
    
        return False
    
    def _check_truncated(self):
        max_steps = 50 + 10 * self.path_removed
        if self.current_step >= max_steps:
            return True
        if self._has_global_guidance() == 1:
            return True
        
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
        if action == 0 and x > 0:  # Down
            x -= 1
        elif action == 1 and x < self.height - 1:  # Up
            x += 1
        elif action == 2 and y > 0:  # Left
            y -= 1
        elif action == 3 and y < self.width - 1:  # Right
            y += 1
        elif action == 4:
            return self.current_position

        next_position = (x, y)
        if self.static_obstacles[next_position] == 0:  # check collision
            self.current_position = next_position  # update
        return self.current_position

    def _calculate_reward(self, current_position, next_position):
        # Calculate reward
        r1, r2, r3 = -0.01, -0.1, 0.1
        
        if next_position in self.global_path and current_position in self.global_path:
        # A large positive reward r 1 + N e × r 3 when the robot reaches one of the cells on the global guidance path, 
        # where r 3 > |r 1 | > 0 and Ne is the number of cells removed from the global guidance path, 
        # between the point where the robot first left that path, to the point where it rejoins it.
            index_current = self.global_path.index(current_position)
            index_next = self.global_path.index(next_position)
            path_removed = abs(index_next - index_current) - 1
            reward = r1 + (path_removed * r3)
        elif self.static_obstacles[next_position[0], next_position[1]] == 1 or next_position in self.dynamic_obstacles.get_positions():
        # A large negative reward r 1 + r 2 when the robot conflicts with a static obstacle or a dynamic obstacle
            reward = r1 + r2
        elif next_position == self.goal:
            # found the goal
            reward = 1
        else:
             #A small negative reward r 1 < 0 when the robot reaches a free
            # cell which is not located on the global guidance
            reward = r1 

        return reward
    
    def _update_path_tracking(self):
        if self.current_position in self.global_path:
            current_index = self.global_path.index(self.current_position)
        if self.last_path_index != -1:  # agent on the global path
            self.path_removed = abs(current_index - self.last_path_index) - 1
            self.last_path_index = current_index
        else:
            self.last_path_index = -1  # not on
            self.path_removed += 1  # accumulate the steps 


    def _get_observation(self):
        """ observation = np.zeros((self.height, self.width, 3), dtype=np.uint8)

        # white for free cells and green for static obstacles
        for i in range(self.height):
            for j in range(self.width):
                if self.static_obstacles[i, j] == 0:
                    observation[i, j] = [255, 255, 255]
                elif self.static_obstacles[i, j] == 1:
                    observation[i, j] = [0, 255, 0]
        
        # black for global path
        for (i, j) in self.global_path:
            observation[i, j] = [0, 0, 0]

        # yellow for dynamic obstacles
        for (i, j) in self.dynamic_obstacles.get_positions():
            observation[i, j] = [255, 255, 0]

        # blue for current position
        observation[self.current_position[0], self.current_position[1]] = [0, 0, 255]
        # red for goal position
        observation[self.goal[0], self.goal[1]] = [255, 0, 0] 
     """
        rgb_observation = self._get_local_observation()  # (fov_size, fov_size, 3)
        guidance = self._global_guidance()               # (fov_size, fov_size)
        guidance = np.expand_dims(guidance, axis=2)       # (fov_size, fov_size, 1)
        observation = np.concatenate((rgb_observation, guidance), axis=2)  # (fov_size, fov_size, 4)
        
        return observation
    
    def _get_local_observation(self):
        local_obs = np.zeros((self.fov_size, self.fov_size, 3), dtype=np.uint8)
        half_fov_h = self.fov_size // 2
        top, left = max(0, self.current_position[0] - half_fov_h), max(0, self.current_position[1] - half_fov_h)
        bottom, right = min(self.height, self.current_position[0] + half_fov_h + 1), min(self.width, self.current_position[1] + half_fov_h + 1)

        for i in range(top, bottom):
            for j in range(left, right):
                local_row, local_col = i - top, j - left
                
                if self.static_obstacles[i, j] == 0:
                    local_obs[local_row, local_col] = [255, 255, 255]  # white for free cells
                elif self.static_obstacles[i, j] == 1:
                    local_obs[local_row, local_col] = [0, 255, 0]  # green for static obstacles
                
                if (i, j) in self.dynamic_obstacles.get_positions():
                    local_obs[local_row, local_col] = [255, 255, 0]  # yellow for dynamic obstacles

                if (i, j) == self.current_position:
                    local_obs[local_row, local_col]  = [0, 0, 255] # blue for agent

        return local_obs
    
    def _global_guidance(self):
        guidance = np.zeros((self.fov_size, self.fov_size), dtype=np.uint8)
        half_fov_h = self.fov_size // 2
        top, left = max(0, self.current_position[0] - half_fov_h), max(0, self.current_position[1] - half_fov_h)
        bottom, right = min(self.height, self.current_position[0] + half_fov_h + 1), min(self.width, self.current_position[1] + half_fov_h + 1)
        
        for i in range(top, bottom):
            for j in range(left, right):
                local_row, local_col = i - top, j - left

                if (i, j) in self.global_path:
                    guidance[local_row, local_col] = 0  # black for global path

        return guidance


    """    def render(self, mode='human'):
        if mode == 'human':
            plt.imshow(self._get_observation())
            plt.axis('on')
        elif mode == 'rgb_array':
            return self._get_observation() """

    
    def render(self):
        if self.render_mode == 'rgb_array':
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
        self.clock.tick(1)  # fps

  

    def close(self):
        pygame.quit()

