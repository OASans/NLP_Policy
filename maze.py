def solve_maze(maze: list[list]) -> bool:
    # 先遍历把enimy能看到的地方转成墙
    # 遍历的过程中同时记录A的初始位置pos: tuple(x, y)
    def DFS(maze, pos, end):
        def legal_step(maze, new_pos):
            # 墙
            if maze[new_pos[0]][new_pos[1]] == 'X':
                return False
            # 边界
            if new_pos[0] < 0 or new_pos[0] >= len(maze):
                return False
            if new_pos[1] < 0 or new_pos[1] >= len(maze[0]):
                return False
            return True

        if pos == end:
            return True
        new_pos = [(pos[0] + 1, pos[1]), (pos[0] - 1, pos[1]), (pos[0], pos[1] + 1), (pos[0], pos[1] - 1)]
        for new in new_pos:
            if legal_step(maze, new) and DFS(maze, new, end):
                return True
        return False

    end = (len(maze) - 1, len(maze[0]) - 1)
    pos = (0, 0)
    return DFS(maze, pos, end)
