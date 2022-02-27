import math
import numpy as np
from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt
from xml.dom.minidom import parse
import xml.dom.minidom


def eulerAnglesToRotationMatrix(theta):

    R_x = np.array([[1,                  0,                   0],
                    [0, math.cos(theta[0]), -math.sin(theta[0])],
                    [0, math.sin(theta[0]),  math.cos(theta[0])]
                    ])

    R_y = np.array([[math.cos(theta[1]),  0, math.sin(theta[1])],
                    [0,                   1,                  0],
                    [-math.sin(theta[1]), 0, math.cos(theta[1])]
                    ])

    R_z = np.array([[math.cos(theta[2]), -math.sin(theta[2]), 0],
                    [math.sin(theta[2]),  math.cos(theta[2]), 0],
                    [0,                   0,                  1]
                    ])

    R = np.dot(R_z, np.dot(R_y, R_x))

    return R


if __name__ == '__main__':
    roll = 0
    pitch = 0
    yaw = 0
    x = 0
    y = 0
    z = 0

    DOMTree = xml.dom.minidom.parse("test.xml")
    livox = DOMTree.documentElement
    devices = livox.getElementsByTagName("Device")
    for device in devices:
        if device.hasAttribute("roll"):
            roll = device.getAttribute("roll")
        if device.hasAttribute("pitch"):
            pitch = device.getAttribute("pitch")
        if device.hasAttribute("yaw"):
            yaw = device.getAttribute("yaw")
        if device.hasAttribute("x"):
            x = device.getAttribute("x")
        if device.hasAttribute("y"):
            y = device.getAttribute("y")
        if device.hasAttribute("z"):
            z = device.getAttribute("z")
    print roll, pitch, yaw, x, y, z

    angles = [float(roll) / 180 * math.pi, float(pitch) / 180 * math.pi, float(yaw) / 180 * math.pi]
    translate = np.array([float(x), float(y), float(z)])

    R = eulerAnglesToRotationMatrix(angles)

    points = np.random.rand(1000, 3)
    points = np.dot(points, R)
    points = points + translate

    # Creating figure
    fig = plt.figure(figsize=(10, 7))
    ax = plt.axes(projection="3d")

    # Creating plot
    ax.scatter3D(points[:, 0], points[:, 1], points[:, 2], color="green")
    plt.title("simple 3D scatter plot")

    # show plot
    plt.show()
