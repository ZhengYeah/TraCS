def gps_to_unit_square(gps, epsilon):
    """
    the transformation from GPS to unit square
    :param gps: the GPS coordinate
    :param epsilon: privacy budget
    :return:
    """
    x, y = gps
    x = kRR(x, [0, 1], epsilon)
    y = kRR(y, [0, 1], epsilon)
    return x, y


def unit_square_to_gps(unit_square, epsilon):
    """
    the transformation from unit square to GPS
    :param unit_square: the unit square coordinate
    :param epsilon: privacy budget
    :return:
    """
    x, y = unit_square
    x = kRR(x, [0, 1], epsilon)
    y = kRR(y, [0, 1], epsilon)
    return x, y
