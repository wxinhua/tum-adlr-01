import heapq  # 导入heapq模块，用于实现优先级队列

# A*算法类
class AStar:
    def __init__(self, grid_height, grid_width, start, goal, static_obstacles):
        # 初始化A*算法
        self.H = grid_height  # 网格高度
        self.W = grid_width  # 网格宽度
        self.start = start  # 起始点
        self.goal = goal  # 目标点
        self.static_obstacles = static_obstacles  # 静态障碍物矩阵

    def heuristic(self, a, b):
        # 计算启发式估计值，这里使用曼哈顿距离
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

    def is_valid(self, node):
        # 检查节点是否有效（在网格范围内且没有障碍物）
        x, y = node
        return 0 <= x < self.H and 0 <= y < self.W and not self.static_obstacles[x, y]

    def a_star_search(self):
        # 实现A*搜索算法
        open_list = []  # 创建一个空的开放列表
        heapq.heappush(open_list, (0, self.start))  # 将起始点加入开放列表
        came_from = {self.start: None}  # 记录路径
        cost_so_far = {self.start: 0}  # 记录从起点到当前节点的成本

        while open_list:
            # 当开放列表不为空时，继续搜索
            _, current = heapq.heappop(open_list)  # 从开放列表中取出具有最小优先级的节点

            if current == self.goal:
                # 如果当前节点是目标节点，则返回路径
                return self.reconstruct_path(came_from)

            for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                # 遍历当前节点的四个邻居节点（上、下、左、右）
                next_node = (current[0] + dx, current[1] + dy)
                if self.is_valid(next_node):
                    # 如果邻居节点是有效的
                    new_cost = cost_so_far[current] + 1  # 计算到邻居节点的新成本
                    if next_node not in cost_so_far or new_cost < cost_so_far[next_node]:
                        # 如果邻居节点不在成本记录中或新成本小于已有成本
                        cost_so_far[next_node] = new_cost  # 更新成本
                        priority = new_cost + self.heuristic(self.goal, next_node)  # 计算优先级
                        heapq.heappush(open_list, (priority, next_node))  # 将邻居节点加入开放列表
                        came_from[next_node] = current  # 记录路径

        return []  # 如果没有找到路径，返回空列表

    def reconstruct_path(self, came_from):
        # 重建路径
        path = []
        current = self.goal
        while current:
            # 从目标点回溯到起点
            path.append(current)
            current = came_from[current]
        path.reverse()  # 反转路径，使其从起点到目标点
        return path
