import static_environment
import Astar
import pickle


# Initialize the environment
grid_height = 100
grid_width = 100
obstacles_density = 0.15
num_maps = 100

# Initialize the position of robot
start = (0, 0)
goal = (99, 99)

# Generate the map
maps_and_paths = []
for i in range(num_maps):
    map_gen = static_environment.MapGenerator(grid_height, grid_width, obstacles_density)
    static_obstacles = map_gen.static_obstacles
    astar = Astar.AStar(grid_height, grid_width, start, goal, static_obstacles)
    global_path = astar.a_star_search()
    if global_path:
        maps_and_paths.append((static_obstacles, global_path))
        if i % 1 == 0:
            print(f"Generated {i} maps")

# save map and global path
with open('maps_and_paths.pkl', 'wb') as f:
    pickle.dump(maps_and_paths, f)

print("All maps and paths have been generated and saved.")


# Visualization
#map_gen.visualize(start, goal, path)
