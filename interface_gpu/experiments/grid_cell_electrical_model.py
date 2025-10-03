# wrap weights funtion around axis in a circular fashion
# https://www.mathworks.com/matlabcentral/answers/1716995-revolving-a-curve-about-the-y-axis-to-generate-a-3d-surface

def torodial_dist(x1, y1, x2, y2, n):
    dx = abs(x2 - x1)
    dy = abs(y2 - y1)

    if dx > n / 2:
        dx = n - dx

    if dy > n / 2:
        dy = n - dy

    return np.sqrt(dx**2 + dy**2)

sigmoid_second_derivative = lambda x: -1 * ((np.exp(x) * (np.exp(x) - 1)) / (np.exp(x) + 1) ** 3)

sigmoid_second_derivative(torodial_dist(x1, y1, x2, y2, n))
