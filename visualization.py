import pickle
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np

# load map and global path
with open('maps_and_paths.pkl', 'rb') as f:
    maps_and_paths = pickle.load(f)


class MapGenerator:
    def __init__(self, height, width, static_obstacles):
        self.height = height
        self.width = width
        self.static_obstacles = static_obstacles

    # visualize the static map
    def visualize(self, start, goal, path=None):
        fig, ax = plt.subplots(figsize=(10, 10))
        ax.set_xlim(0, self.width)
        ax.set_ylim(0, self.height)
        ax.set_xticks(np.arange(0, self.width, 10))
        ax.set_yticks(np.arange(0, self.height, 10))
        ax.grid(True)

        for i in range(self.height):
            for j in range(self.width):
                if self.static_obstacles[i, j]:
                    ax.add_patch(plt.Rectangle((j, self.height - i - 1), 1, 1, color='black'))

        start_patch = plt.Circle((start[1] + 0.5, self.height - start[0] - 0.5), 0.4, color='blue')
        goal_patch = plt.Circle((goal[1] + 0.5, self.height - goal[0] - 0.5), 0.4, color='red')
        ax.add_patch(start_patch)
        ax.add_patch(goal_patch)

        if path:
            for (x, y) in path:
                ax.add_patch(plt.Rectangle((y, self.height - x - 1), 1, 1, color='green', alpha=0.5))

        plt.show()

    # visualize the dynamic map
    def visualize_with_dynamic_obstacles(static_obstacles, start, goal, path, dynamic_obstacles, num_steps=50):
        fig, ax = plt.subplots(figsize=(10, 10))
        ax.set_xlim(0, static_obstacles.shape[1])
        ax.set_ylim(0, static_obstacles.shape[0])
        ax.set_xticks(np.arange(0, static_obstacles.shape[1], 10))
        ax.set_yticks(np.arange(0, static_obstacles.shape[0], 10))
        ax.grid(True)

        for i in range(static_obstacles.shape[0]):
            for j in range(static_obstacles.shape[1]):
                if static_obstacles[i, j]:
                    ax.add_patch(plt.Rectangle((j, static_obstacles.shape[0] - i - 1), 1, 1, color='black'))

        start_patch = plt.Circle((start[1] + 0.5, static_obstacles.shape[0] - start[0] - 0.5), 0.4, color='blue')
        goal_patch = plt.Circle((goal[1] + 0.5, static_obstacles.shape[0] - goal[0] - 0.5), 0.4, color='red')
        ax.add_patch(start_patch)
        ax.add_patch(goal_patch)

        if path:
            for (x, y) in path:
                ax.add_patch(plt.Rectangle((y, static_obstacles.shape[0] - x - 1), 1, 1, color='green', alpha=0.5))

        dynamic_patches = []
        for (x, y) in dynamic_obstacles.get_positions():
            patch = plt.Rectangle((y, static_obstacles.shape[0] - x - 1), 1, 1, color='yellow')
            dynamic_patches.append(patch)
            ax.add_patch(patch)

        def update(t):
            dynamic_obstacles.update_positions()
            for patch in dynamic_patches:
                patch.remove()
            dynamic_patches.clear()

            for (x, y) in dynamic_obstacles.get_positions():
                patch = plt.Rectangle((y, static_obstacles.shape[0] - x - 1), 1, 1, color='yellow')
                dynamic_patches.append(patch)
                ax.add_patch(patch)

        ani = animation.FuncAnimation(fig, update, frames=num_steps, repeat=True)
        plt.show()


# Check Ith map and global path
#static_obstacles, global_path = maps_and_paths[70]
#map_gen = MapGenerator(100, 100, static_obstacles)
#map_gen.visualize((0, 0), (99, 99), global_path)


# Visualize all maps and global paths
#for i, (static_obstacles, global_path) in enumerate(maps_and_paths):
    #print(f"Visualizing map {i+1}")
    #map_gen = MapGenerator(100, 100, static_obstacles)
    #map_gen.visualize((0, 0), (99, 99), global_path)
