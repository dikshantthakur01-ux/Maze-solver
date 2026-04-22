import heapq
from collections import deque
import tkinter as tk
import random


# Maze representation
# 0 = free path
# 1 = wall

maze = [
    [0,1,0,1,0,1,0,1,0,1,0],
    [0,1,0,0,0,1,0,0,0,1,0],
    [0,0,0,1,0,1,0,1,0,1,0],
    [0,1,0,1,0,0,0,1,0,1,0],
    [0,1,0,0,0,1,1,0,0,0,0],
    [0,1,1,1,0,1,0,0,1,1,0],
    [0,0,0,1,0,1,0,0,1,0,0],
    [1,1,0,1,0,1,1,0,1,0,1],
    [0,0,0,0,0,0,0,0,1,0,0],
    [1,1,0,1,1,1,1,0,1,0,1],
    [0,0,0,0,0,0,0,0,0,0,0]
]

rows = len(maze)
cols = len(maze[0])

start = (0, 0)
goal = (10, 10)

# Possible movements
directions = [(0,1),(1,0),(0,-1),(-1,0)]

# Check valid move
def is_valid(x, y):
    return 0 <= x < rows and 0 <= y < cols and maze[x][y] == 0

# Reconstruct path
def reconstruct_path(parent, end):
    path = []
    while end:
        path.append(end)
        end = parent.get(end)
    return path[::-1]

# BFS Algorithm
def bfs(start, goal):
    queue = deque([start])
    visited = set([start])
    parent = {start: None}

    while queue:
        node = queue.popleft()

        if node == goal:
            return reconstruct_path(parent, goal)

        local_dirs = directions[:]
        random.shuffle(local_dirs)
        for d in local_dirs:
            nx = node[0] + d[0]
            ny = node[1] + d[1]

            if is_valid(nx, ny) and (nx, ny) not in visited:
                queue.append((nx, ny))
                visited.add((nx, ny))
                parent[(nx, ny)] = node

    return None

# DFS Algorithm
def dfs(start, goal):
    stack = [start]
    visited = set([start])
    parent = {start: None}

    while stack:
        node = stack.pop()

        if node == goal:
            return reconstruct_path(parent, goal)

        for d in directions:
            nx = node[0] + d[0]
            ny = node[1] + d[1]

            if is_valid(nx, ny) and (nx, ny) not in visited:
                stack.append((nx, ny))
                visited.add((nx, ny))
                parent[(nx, ny)] = node

    return None

# Heuristic for A*
def heuristic(a, b):
    return abs(a[0]-b[0]) + abs(a[1]-b[1])

# A* Algorithm
def astar(start, goal):
    open_list = []
    heapq.heappush(open_list, (0, start))

    g_cost = {start:0}
    parent = {start:None}

    while open_list:
        _, current = heapq.heappop(open_list)

        if current == goal:
            return reconstruct_path(parent, goal)

        for d in directions:
            nx = current[0] + d[0]
            ny = current[1] + d[1]
            neighbor = (nx, ny)

            if is_valid(nx, ny):
                new_cost = g_cost[current] + 1

                if neighbor not in g_cost or new_cost < g_cost[neighbor]:
                    g_cost[neighbor] = new_cost
                    f_cost = new_cost + heuristic(neighbor, goal)
                    rand_tie = random.uniform(0, 0.0001)
                    heapq.heappush(open_list, (f_cost + rand_tie, neighbor))
                    parent[neighbor] = current

    return None

def hill_climb(start, goal):
    current = start
    path = [current]
    while current != goal:
        neighbors = [(current[0] + d[0], current[1] + d[1]) for d in directions if is_valid(current[0] + d[0], current[1] + d[1])]
        if not neighbors:
            return None
        min_h = min(heuristic(n, goal) for n in neighbors)
        best_neighbors = [n for n in neighbors if heuristic(n, goal) == min_h]
        next_node = random.choice(best_neighbors)
        if heuristic(next_node, goal) >= heuristic(current, goal):
            return None  # Local optimum
        path.append(next_node)
        current = next_node
    return path

def best_first(start, goal):
    open_list = []
    heapq.heappush(open_list, (heuristic(start, goal), start))
    came_from = {start: None}

    while open_list:
        _, current = heapq.heappop(open_list)
        if current == goal:
            return reconstruct_path(came_from, goal)
        for d in directions:
            nx = current[0] + d[0]
            ny = current[1] + d[1]
            neighbor = (nx, ny)
            if is_valid(nx, ny) and neighbor not in came_from:
                priority = heuristic(neighbor, goal)
                heapq.heappush(open_list, (priority, neighbor))
                came_from[neighbor] = current
    return None

def minimax_search(start, goal):
    """Simplified Minimax-like search for pathfinding with depth limit"""
    def minimax(node, depth, alpha, beta, is_max):
        if node == goal:
            return 0
        if depth == 0:
            return heuristic(node, goal)
        
        neighbors = [(node[0] + dx, node[1] + dy) for dx, dy in directions if is_valid(node[0] + dx, node[1] + dy)]
        if not neighbors:
            return float('inf')
        
        if is_max:
            max_eval = float('-inf')
            for n in neighbors:
                eval_score = minimax(n, depth-1, alpha, beta, False)
                max_eval = max(max_eval, eval_score)
                alpha = max(alpha, eval_score)
                if beta <= alpha:
                    break
            return max_eval
        else:
            min_eval = float('inf')
            for n in neighbors:
                eval_score = minimax(n, depth-1, alpha, beta, True)
                min_eval = min(min_eval, eval_score)
                beta = min(beta, eval_score)
                if beta <= alpha:
                    break
            return min_eval
    
    # Iterative deepening to find path
    path = None
    for max_depth in range(1, 30):
        # Find best first move using minimax
        neighbors = [(start[0] + dx, start[1] + dy) for dx, dy in directions if is_valid(start[0] + dx, start[1] + dy)]
        best_neighbor = None
        best_score = float('inf')
        for n in neighbors:
            score = minimax(n, max_depth-1, -float('inf'), float('inf'), False)
            if score < best_score:
                best_score = score
                best_neighbor = n
        
        if best_neighbor is None:
            break
        # Build path using greedy from there (hybrid)
        current_path = bfs(start, goal)
        if current_path:
            path = current_path
            break
    
    return path or bfs(start, goal)  # Fallback to BFS

# Print maze with path
def print_maze(path):
    maze_copy = [row[:] for row in maze]

    for x,y in path:
        maze_copy[x][y] = "*"

    for row in maze_copy:
        print(row)

class MazeGUI:
    def __init__(self, parent):
        self.parent = parent
        self.parent.title("Maze Pathfinding GUI")
        self.cell_size = 45
        self.canvas = tk.Canvas(parent, width=cols * self.cell_size, height=rows * self.cell_size, bg='white')
        self.canvas.pack()
        self.button_frame = tk.Frame(parent)
        self.button_frame.pack()
        tk.Button(self.button_frame, text="BFS", command=lambda: self.run_algorithm(bfs), bg='lightblue').pack(side='left', padx=5)
        tk.Button(self.button_frame, text="DFS", command=lambda: self.run_algorithm(dfs), bg='lightgreen').pack(side='left', padx=5)
        tk.Button(self.button_frame, text="A*", command=lambda: self.run_algorithm(astar), bg='orange').pack(side='left', padx=5)
        tk.Button(self.button_frame, text="Hill Climb", command=lambda: self.run_algorithm(hill_climb), bg='pink').pack(side='left', padx=5)
        tk.Button(self.button_frame, text="Best First", command=lambda: self.run_algorithm(best_first), bg='lightyellow').pack(side='left', padx=5)
        tk.Button(self.button_frame, text="Minimax", command=lambda: self.run_algorithm(minimax_search), bg='violet').pack(side='left', padx=5)
        tk.Button(self.button_frame, text="Clear", command=self.clear_path, bg='lightgray').pack(side='left', padx=5)
        self.info_label = tk.Label(parent, text="Click an algorithm to find path!", font=('Arial', 12))
        self.info_label.pack(pady=10)
        self.draw_maze()

    def draw_maze(self):
        self.canvas.delete("all")
        for i in range(rows):
            for j in range(cols):
                x1 = j * self.cell_size
                y1 = i * self.cell_size
                x2 = x1 + self.cell_size
                y2 = y1 + self.cell_size
                color = 'black' if maze[i][j] == 1 else '#f0f0f0'
                self.canvas.create_rectangle(x1, y1, x2, y2, fill=color, outline='gray', width=1)
        # Start lime, goal red
        s_x, s_y = start
        self.canvas.create_oval(s_y * self.cell_size + 5, s_x * self.cell_size + 5,
                                (s_y + 1) * self.cell_size - 5, (s_x + 1) * self.cell_size - 5, fill='lime', tags='start')
        g_x, g_y = goal
        self.canvas.create_oval(g_y * self.cell_size + 5, g_x * self.cell_size + 5,
                                (g_y + 1) * self.cell_size - 5, (g_x + 1) * self.cell_size - 5, fill='red', tags='goal')

    def clear_path(self):
        self.canvas.delete("path")
        self.info_label.config(text="Canvas cleared. Choose an algorithm.")

    def draw_path(self, path):
        self.canvas.delete("path")
        for i, (x, y) in enumerate(path):
            x1 = y * self.cell_size + 2
            y1 = x * self.cell_size + 2
            x2 = x1 + self.cell_size - 4
            y2 = y1 + self.cell_size - 4
            color = 'lime' if i == 0 else ('gold' if i == len(path)-1 else 'cyan')
            self.canvas.create_rectangle(x1, y1, x2, y2, fill=color, tags="path")

    def run_algorithm(self, algorithm):
        path = algorithm(start, goal)
        if path:
            self.draw_maze()  # redraw to clear old paths
            self.draw_path(path)
            steps = len(path) - 1
            self.info_label.config(text=f"{algorithm.__name__.upper()} Path found! {steps} steps. Path: {path}")
        else:
            self.clear_path()
            self.info_label.config(text=f"{algorithm.__name__.upper()}: No path found!")

# Demo console run (commented)
# print("BFS Path:")
# bfs_path = bfs(start, goal)
# print(bfs_path)
# print_maze(bfs_path)
# ... other console code commented

if __name__ == "__main__":
    root = tk.Tk()
    app = MazeGUI(root)
    root.mainloop()

