def gps_to_unit_square(x, y, x_min: float, x_max: float, y_min: float, y_max: float):
    """
    transform GPS coordinates to unit-square coordinates
    """
    assert len(x) == len(y)
    x_length = x_max - x_min
    y_length = y_max - y_min
    return (x - x_min) / x_length, (y - y_min) / y_length


def unit_square_to_gps(traj, x_min: float, x_max: float, y_min: float, y_max: float):
    """
    transform unit square coordinates to GPS coordinates
    """
    x_length = x_max - x_min
    y_length = y_max - y_min
    for i in range(len(traj)):
        traj[i] = traj[i][0] * x_length + x_min, traj[i][1] * y_length + y_min
    return traj
