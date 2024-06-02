import heapq


class AStar:
    def __init__(self, grid_height, grid_width, start, goal, static_obstacles):
        self.H = grid_height
        self.W = grid_width
        self.start = start
        self.goal = goal
        self.static_obstacles = static_obstacles

    def heuristic(self, a, b):
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

    def is_valid(self, node):
        x, y = node
        return 0 <= x < self.H and 0 <= y < self.W and not self.static_obstacles[x, y]

    def a_star_search(self):
        open_list = []
        heapq.heappush(open_list, (0, self.start))
        came_from = {self.start: None}
        cost_so_far = {self.start: 0}

        while open_list:
            _, current = heapq.heappop(open_list)

            if current == self.goal:
                return self.reconstruct_path(came_from)

            for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                next_node = (current[0] + dx, current[1] + dy)
                if self.is_valid(next_node):
                    new_cost = cost_so_far[current] + 1
                    if next_node not in cost_so_far or new_cost < cost_so_far[next_node]:
                        cost_so_far[next_node] = new_cost
                        priority = new_cost + self.heuristic(self.goal, next_node)
                        heapq.heappush(open_list, (priority, next_node))
                        came_from[next_node] = current

        return []



    def reconstruct_path(self, came_from):
        path = []
        current = self.goal
        while current:
            path.append(current)
            current = came_from[current]
        path.reverse()
        return path
