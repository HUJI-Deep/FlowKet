import numpy

from .moves import right, down, down_left, up_right


def zigzag(input_size):
    if len(input_size) != 2 or input_size[0] != input_size[1]:
        raise ValueError("Zigzag order only supports 2D squares.")
    total_size = numpy.prod(input_size)
    order = [(0, 0), (0, 1)]
    move = right
    while len(order) < total_size:
        y, x = order[-1]
        if move == right:
            move = down_left if y + 1 < input_size[0] else up_right
        elif move == down:
            move = up_right if x + 1 < input_size[1] else down_left
        if move == down_left:
            can_down = y + 1 < input_size[0]
            move = down_left if x > 0 and can_down else down
            if not can_down:
                move = right
        if move == up_right:
            can_right = x + 1 < input_size[1]
            move = up_right if can_right and y > 0 else down
            if not can_right:
                move = down
        order.append(move(x, y))
    return order
