from enum import Enum
from random import randrange as rand, choice

T = [[1, 1, 1],
     [0, 1, 0]]
S = [[0, 2, 2],
     [2, 2, 0]]
Z = [[3, 3, 0],
     [0, 3, 3]]
J = [[4, 0, 0],
     [4, 4, 4]]
L = [[0, 0, 5],
     [5, 5, 5]]
I = [[6, 6, 6, 6]]
O = [[7, 7],
     [7, 7]]
TETROMINOES = [T, S, Z, J, L, I, O]

W = 10
H = 22


class Actions(Enum):
    LEFT = 'LEFT'
    RIGHT = 'RIGHT'
    ROTATE = 'ROTATE'
    DROP_ONE = 'DROP_ONE'
    DROP_TO_BOTTOM = 'DROP_TO_BOTTOM'


def rotate_clockwise(tetromino):
    return [
        [
            tetromino[y][x] for y in range(len(tetromino))
        ] for x in range(len(tetromino[0]) - 1, -1, -1)
    ]


def check_collision(board, tetromino, offset):
    off_x, off_y = offset
    for y, row in enumerate(tetromino):
        for x, v in enumerate(row):
            try:
                if v and board[y + off_y][x + off_x]:
                    return True
            except IndexError:
                return True
    return False


def remove_row(board, row):
    w = len(board[row])
    del board[row]
    return [[0 for _ in range(w)]] + board


def join_matrices(m1, m2, m_offset):
    off_x, off_y = m_offset
    for y, row in enumerate(m2):
        for x, v in enumerate(row):
            m1[y + off_y - 1][x + off_x] += v
    return m1


def new_board(w, h):
    board = [[0 for _ in range(w)] for _ in range(h)]
    board += [[1 for _ in range(w)]]
    return board


class Tetris(object):
    def __init__(self, w=W, h=H, tetrominoes=TETROMINOES):
        self.w = w
        self.h = h
        self.tetrominoes = tetrominoes
        self.next_tetromino = choice(self.tetrominoes)
        self.board = new_board(w, h)
        self.new_tetromino()
        self.level = 1
        self.score = 0
        self.lines = 0
        self.game_over = False

    @property
    def tetromino_h(self):
        return len(self.tetromino)

    @property
    def tetromino_w(self):
        return len(self.tetromino[0])

    def new_tetromino(self):
        self.tetromino = self.next_tetromino[:]
        self.next_tetromino = choice(self.tetrominoes)
        self.tetromino_x = int(self.w / 2 - self.tetromino_w / 2)
        self.tetromino_y = 0

        if check_collision(self.board, self.tetromino, (self.tetromino_x, self.tetromino_y)):
            self.game_over = True

    def add_cl_lines(self, n):
        linescores = [0, 40, 100, 300, 1200]
        self.lines += n
        self.score += linescores[n] * self.level
        if self.lines >= self.level * 6:
            self.level += 1

    def move(self, delta_x):
        if not self.game_over:
            new_x = self.tetromino_x + delta_x
            if new_x < 0:
                new_x = 0
            if new_x > self.w - self.tetromino_w:
                new_x = self.h - self.tetromino_w
            if not check_collision(self.board, self.tetromino, (new_x, self.tetromino_y)):
                self.tetromino_x = new_x

    def drop(self, manual=False):
        self.score += 1 if manual else 0
        self.tetromino_y += 1

        if check_collision(self.board, self.tetromino, (self.tetromino_x, self.tetromino_y)):
            self.board = join_matrices(
                self.board,
                self.tetromino,
                (self.tetromino_x, self.tetromino_y))
            self.new_tetromino()
            cleared_rows = 0

            while True:
                for i, row in enumerate(self.board[:-1]):
                    if 0 not in row:
                        self.board = remove_row(self.board, i)
                        cleared_rows += 1
                        break
                else:
                    break
            self.add_cl_lines(cleared_rows)
            return True

        return False

    def insta_drop(self):
        while not self.drop(manual=True):
            pass

    def rotate_tetromino(self):
        new_tetromino = rotate_clockwise(self.tetromino)
        if not check_collision(self.board, new_tetromino, (self.tetromino_x, self.tetromino_y)):
            self.tetromino = new_tetromino

    def action(self, which):
        return {
            Actions.LEFT: lambda: self.move(-1),
            Actions.RIGHT: lambda: self.move(1),
            Actions.ROTATE: lambda: self.rotate_tetromino(),
            Actions.DROP_ONE: lambda: self.drop(True),
            Actions.DROP_TO_BOTTOM: lambda: self.insta_drop(),
        }[which]()

    def get_row(self, y):
        return self.get_board_with_tetromino()[y]

    def get_col(self, x):
        return [self.get_board_with_tetromino()[y][x] for y in range(self.h)]

    def get_board_with_tetromino(self):
        b = [[v for v in row] for row in self.board]

        for y in range(self.tetromino_h):
            for x in range(self.tetromino_w):
                if self.tetromino[y][x] != 0:
                    b[self.tetromino_y + y][self.tetromino_x + x] = self.tetromino[y][x]

        return b

    def __str__(self):
        l = lambda v: ' ' if v == 0 else str(v)
        return '|' + '|\n|'.join(''.join(l(v) for v in row) for row in self.get_board_with_tetromino()) + '|'
