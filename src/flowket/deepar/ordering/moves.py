def up(x, y):
    return y - 1, x


def down(x, y):
    return y + 1, x


def left(x, y):
    return y, x - 1


def right(x, y):
    return y, x + 1


def up_right(x, y):
    return y - 1, x + 1


def up_left(x, y):
    return y - 1, x - 1


def down_right(x, y):
    return y + 1, x + 1


def down_left(x, y):
    return y + 1, x - 1
