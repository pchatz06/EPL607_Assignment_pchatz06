from PIL import Image, ImageDraw
import random

"""
This function is the most important function for this assignment (checks which pixels to draw basically) here is the explanation:
With this function I check if a given point (px, py) is inside the triangle.
- every triangle has 3 edges (lines connecting its 3 points)
- A point can either be on one side of the edge, on the other size of the edge or exactly on the edge
- we basically determine which side the point is on using the sign function in the code (inside the function)
- A point inside the triangle must be on the same side for all three edges
- If it is on a different side for at least one edge, then it is outside the triangle
- for each of the edges, we calculate the sign of the point using the formula:
- (x1 - x3) * (y2 - y3) - (x2 - x3) * (y1 - y3) # (V1-V2, V2-V3, V3-V1)
- where x1, y1 represents the point, and the (x2, y2) and (x3, y3) represents the checked edge
- So basically, each edge divides the space into 2 sides, if a point is on the same side for all three edges, its inside
- the triangle, if a point is on different sides of the edges, its outside.


So:
- If the point is on the same side of all three edges it is inside
- If the point is on different sides is outside.
- This function check this, by calculating the position of a point to a line.
- A positive or negative result tell us which side the point is on according to an edge.
- If the point has 3 similar signs for the 3 edges (3 positive or 3 negative) then it is inside,
- otherwise it is outside.

- To be exact, if the result is:
- Positive, the point is on one side of the line
- Negative, the point is on the other side
- zero, the point is EXACTLY on the line. 

The function returns true if the point has all signs the same thus the point is inside, and false, if the point is outside,
since it has different signs relative to the edges.
"""

from PIL import Image, ImageDraw
import random


def is_inside_triangle(px, py, p1, p2, p3):
    def sign(pc, p1, p2):
        return (pc[0] - p2[0]) * (p1[1] - p2[1]) - (p1[0] - p2[0]) * (pc[1] - p2[1])

    e1 = sign((px, py), p1, p2)
    e2 = sign((px, py), p2, p3)
    e3 = sign((px, py), p3, p1)

    has_neg = (e1 < 0) or (e2 < 0) or (e3 < 0)
    has_pos = (e1 > 0) or (e2 > 0) or (e3 > 0)

    return not (has_neg and has_pos)
# This function calculates the equation of a plane given three points in 3D space
# The three points (v1, v2, v3) define the plane, and the output is the plane equation in the form:
# Ax + By + Cz + D = 0 where (A, B, C) is the normal vector and D is a constant.

"""
Here I calculate the equation of the plane with 3 points.
the three points of vectors (v1, v2, v3) define a plane, and the output is the plane equation in the form:
Ax + By + Cz + D = 0 where (A, B, C) is the normal vector and D is a constant.
The reason I do this, is to solve to find z for each pixel in order to identify what triangle has the lowest z to be drawn by its color.
"""
def compute_plane(v1, v2, v3):
    #vectors
    x1, y1, z1 = v1
    x2, y2, z2 = v2
    x3, y3, z3 = v3

    u = (x2 - x1, y2 - y1, z2 - z1)
    v = (x3 - x1, y3 - y1, z3 - z1)

    # Cross product to find normal vector (A, B, C)
    A = u[1] * v[2] - u[2] * v[1]
    B = u[2] * v[0] - u[0] * v[2]
    C = u[0] * v[1] - u[1] * v[0]
    D = -(A * x1 + B * y1 + C * z1)

    return A, B, C, D


def rasterize_triangle(image, zbuffer, v1, v2, v3, color):
    draw = ImageDraw.Draw(image)
    A, B, C, D = compute_plane(v1, v2, v3)

    # whe we will solve with z, it will be divided with C... we have to avoid this case.
    if C == 0:
        return

    # bounding box calcuation, really simple (added the image bound also in the calculations for further performance).
    min_x = round(max(min(v1[0], v2[0], v3[0]), 0))
    max_x = round(min(max(v1[0], v2[0], v3[0]), image.width - 1))
    min_y = round(max(min(v1[1], v2[1], v3[1]), 0))
    max_y = round(min(max(v1[1], v2[1], v3[1]), image.height - 1))

    for x in range(min_x, max_x + 1):
        for y in range(min_y, max_y + 1):
            if not is_inside_triangle(x, y, v1, v2, v3):
                continue

            # Calculate z from plane: z = -(Ax + By + D) / C
            z = -(A * x + B * y + D) / C

            # if the z has lower value that the buffer then a triangle closer to the previous should drawn the pixel
            if z < zbuffer[x][y]:
                zbuffer[x][y] = z
                draw.point((x, y), fill=color)


width, height = 1000, 1000
image = Image.new("RGB", (width, height), (255, 255, 255))

# initially fill the z buffer with inf values, to be drawn by the first triangle that passes through the pixel
zbuffer = [[float('inf')] * height for _ in range(width)]


# these vertices and later on triangles created by 3, create a cube
vertices = [
    (-50, -50, 1),  # Front-bottom-left
    (50, -50, 1),  # Front-bottom-right
    (50, 50, 1),  # Front-top-right
    (-50, 50, 1),  # Front-top-left
    (-50, -50, 2),  # Back-bottom-left
    (150, -50, 2),  # Back-bottom-right
    (150, 150, 2),  # Back-top-right
    (-50, 150, 2)  # Back-top-left
]


# the definition of the triangle is like so [ vector 1, vector 2, vector 3, color RGB)
triangles = [
    [vertices[0], vertices[1], vertices[2], (255, 0, 0)],  # Red triangle
    [vertices[0], vertices[2], vertices[3], (255, 0, 0)],  # Red triangle
    [vertices[4], vertices[5], vertices[6], (0, 255, 0)],  # Green triangle
    [vertices[4], vertices[6], vertices[7], (0, 255, 0)],  # Green triangle
    [vertices[0], vertices[3], vertices[7], (0, 0, 255)],  # Blue triangle
    [vertices[0], vertices[7], vertices[4], (0, 0, 255)],  # Blue triangle
    [vertices[1], vertices[2], vertices[6], (255, 255, 0)],  # Yellow triangle
    [vertices[1], vertices[6], vertices[5], (255, 255, 0)],  # Yellow triangle
    [vertices[2], vertices[3], vertices[7], (255, 165, 0)],  # Orange triangle
    [vertices[2], vertices[7], vertices[6], (255, 165, 0)],  # Orange triangle
    [vertices[0], vertices[1], vertices[5], (255, 255, 255)],  # White triangle
    [vertices[0], vertices[5], vertices[4], (255, 255, 255)],  # White triangle
]

# rasterize triangles, devide their x and y by z, and also, move the (0,0) to the center of the picture using width/2 for x and height / 2 for y like shown below.
for t in triangles:
    p1, p2, p3 = (
        (t[0][0] / t[0][2] + width / 2, height / 2 - t[0][1] / t[0][2], t[0][2]),
        (t[1][0] / t[1][2] + width / 2, height / 2 - t[1][1] / t[1][2], t[1][2]),
        (t[2][0] / t[2][2] + width / 2, height / 2 - t[2][1] / t[2][2], t[2][2])
    )
    color = t[3]
    rasterize_triangle(image, zbuffer, p1, p2, p3, color)

image.save("img.png")
image.show()
