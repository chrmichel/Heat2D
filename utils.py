import numpy as np
from math import exp
from enum import Enum


class Direction(str, Enum):
    UP = "up"
    DOWN = "down"
    LEFT = "left"
    RIGHT = "right"


class Type(str, Enum):
    VALUE = "value"
    NEUMANN = "neumann"
    MIXED = "mixed"


class Condition:
    def __init__(self, type: Type = "value", expr: float = 0.) -> None:
        self.type = type
        self.expr = expr


class Boundary:
    def __init__(
            self, b_up: Condition, b_down: Condition,
            b_left: Condition, b_right: Condition
    ) -> None:
        self.up = b_up
        self.down = b_down
        self.left = b_left
        self.right = b_right


DEFAULT_BOUNDARY = Boundary(
    Condition(type="neumann"),
    Condition(type="neumann"),
    Condition(type="neumann"),
    Condition(type="neumann")
)


def gauss(x, y):
    return np.exp(-0.5*(x**2+y**2)/0.2**2)


def init_cos(x, y):
    return np.cos(2*np.pi*x) * np.cos(2*np.pi*y)


def init_sin(x, y):
    return np.sin(2*np.pi*x) * np.sin(2*np.pi*y)


x = np.linspace(-0.5, 0.5, 100)
y = np.linspace(-0.5, 0.5, 100)
X, Y = np.meshgrid(x, y)

# DEFAULT_INITIAL = 100*np.ones((100, 100))
DEFAULT_INITIAL = init_cos(X, Y)


THERMAL_DICT = {
    "silver": 165.63,
    "copper": 111.,
    "aluminum": 97.,
    "gold": 127.,
    "iron": 23.,
    "pvc": 0.08,
    "brick": 0.52,
    "tin": 40.,
}