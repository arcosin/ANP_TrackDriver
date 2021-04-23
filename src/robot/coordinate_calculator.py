import mpmath
import numpy as np

class coordinateCalculator:
    def __init__(self, l, d):
        self.x1 = 0  # left rear wheel
        self.y1 = 0
        self.x2 = 0  # right rear wheel
        self.y2 = d
        self.l = l  # the length from the midpoint of the two front wheels to the midpoint of the two rear wheels
        self.d = d  # the distance between the two rear wheels

    # make sure your unit of distance in v, l, d, x, y are consistent
    def move(self, alpha, v, t):
        if alpha == 0:
            self.x1 += v*t
            self.x2 += v*t
            return
        reverse_alpha = False
        reverse_v = False
        alpha = mpmath.radians(alpha)
        if alpha < 0:
            alpha = -alpha
            reverse_alpha = True

        if v < 0:
            v = -v
            reverse_v = True

        R = self.l * mpmath.cot(alpha) + 0.5 * self.d  # outer radius
        r = self.l * mpmath.cot(alpha) - 0.5 * self.d  # inner radius
        w = v / R  # angular velocity
        beta = w * t  # angle that has been turned after the move viewed from the center of the circle
        assert(beta < mpmath.pi)        # make sure it still forms a triangle
        theta = 0.5 * beta  # the angle of deviation from the original orientation
        m = 2 * R * mpmath.sin(theta)   # displacement of outer rear wheel
        n = 2 * r * mpmath.sin(theta)   # displacement of inner rear wheel
        if not reverse_alpha and not reverse_v:
            delta_x1, delta_y1 = mpmath.cos(theta) * m, mpmath.sin(theta) * m
            delta_x2, delta_y2 = mpmath.cos(theta) * n, mpmath.sin(theta) * n
        elif reverse_alpha and not reverse_v:
            delta_x2, delta_y2 = mpmath.cos(theta) * m, -mpmath.sin(theta) * m
            delta_x1, delta_y1 = mpmath.cos(theta) * n, -mpmath.sin(theta) * n
        elif not reverse_alpha and reverse_v:
            delta_x1, delta_y1 = -mpmath.cos(theta) * m, mpmath.sin(theta) * m
            delta_x2, delta_y2 = -mpmath.cos(theta) * n, mpmath.sin(theta) * n
        else:
            delta_x2, delta_y2 = -mpmath.cos(theta) * m, -mpmath.sin(theta) * m
            delta_x1, delta_y1 = -mpmath.cos(theta) * n, -mpmath.sin(theta) * n

        delta_x1 = float(delta_x1)
        delta_y1 = float(delta_y1)
        delta_x2 = float(delta_x2)
        delta_y2 = float(delta_y2)

        # the current calculation is based on this relative reference system
        # we need to map this coordinate in the relative reference system back to the absolute system

        # use the dot product of the rear wheel line to find the rotation from the relative reference system to the absolute system
        if self.y1 != 0 or self.y2 != 10:  # no rotation is needed
            bottom = np.sqrt(((self.x2 - self.x1) * (self.x2 - self.x1) + (self.y2 - self.y1) * (self.y2 - self.y1))) * 10
            top = (self.y2 - self.y1) * 10
            cosine_value = float(top / bottom)
            rotation = mpmath.acos(cosine_value)
            # print(rotation)
            if self.x1 < self.x2:   # since arccos does not give direction, we need to do a check here
                rotation = -rotation
            # let A be the linear transformation that maps a relative referenced coordinate to an absolute referenced coordinate
            A = np.asarray([[float(mpmath.cos(rotation)), float(-mpmath.sin(rotation))], [float(mpmath.sin(rotation)), float(mpmath.cos(rotation))]])
            # then we map delta_1 and delta_2 to the absolute reference system
            relative_delta_1 = np.asarray([delta_x1, delta_y1])
            relative_delta_2 = np.asarray([delta_x2, delta_y2])
            absolute_delta_1 = np.matmul(A, relative_delta_1)
            absolute_delta_2 = np.matmul(A, relative_delta_2)
            delta_x1 = absolute_delta_1[0]
            delta_y1 = absolute_delta_1[1]
            delta_x2 = absolute_delta_2[0]
            delta_y2 = absolute_delta_2[1]
        # make vector addition in the absolute reference system
        self.x1 += delta_x1
        self.x2 += delta_x2
        self.y1 += delta_y1
        self.y2 += delta_y2
        dist = mpmath.sqrt((self.x1 - self.x2) * (self.x1 - self.x2) + (self.y2 - self.y1) * (self.y2 - self.y1))
        assert abs(dist - self.d) <= self.d * 1e-4

if __name__ == "__main__":
    print("Testing Coordinate Calculator")
    c = CoordinateCalculator(100, 10)
    print(c.x1, c.y1, c.x2, c.y2)

    c.move(15, 10, 2)
    print("move", 15, 10, 10)
    print(c.x1, c.y1, c.x2, c.y2)

    c.move(15, -10, 2)
    print("move", 15, -10, 10)
    print(c.x1, c.y1, c.x2, c.y2)

    c.move(15, 10, 2)
    print("move", 15, 10, 10)
    print(c.x1, c.y1, c.x2, c.y2)

    c.move(15, -10, 2)
    print("move", 15, -10, 10)
    print(c.x1, c.y1, c.x2, c.y2)

    c.move(15, -10, 2)
    print("move", 15, -10, 10)
    print(c.x1, c.y1, c.x2, c.y2)

    c.move(50, 10, 10)
    print("move", 50, 10, 10)
    print(c.x1, c.y1, c.x2, c.y2)

    c.move(15, 10, 1)
    print("move", 15, 10, 1)
    print(c.x1, c.y1, c.x2, c.y2)

    c.move(-50, -10, 10)
    print("move", -50, -10, 10)
    print(c.x1, c.y1, c.x2, c.y2)

    c.move(50, -10, 10)
    print("move", 50, -10, 10)
    print(c.x1, c.y1, c.x2, c.y2)