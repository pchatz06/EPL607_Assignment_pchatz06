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
import pywavefront


class Camera:
    def __init__(self, position, look_at, up, fov):
        self.position = np.array(position)
        self.look_at = np.array(look_at)
        self.up = np.array(up)
        self.fov = fov

    def world_to_camera(self, point):
        # Create coordinate system
        forward = normalize(self.look_at - self.position)
        right = normalize(np.cross(forward, self.up))
        up = np.cross(right, forward)

        # Transform point to camera space
        rel = np.array(point) - self.position
        x = np.dot(rel, right)
        y = np.dot(rel, up)
        z = np.dot(rel, forward)
        return np.array([x, y, z])


class Material:
    def __init__(self, color, ambient=0.1, diffuse=0.7, specular=0.2,
                 shininess=32):  # ambient + diffuse + specular must be 1 ideally...
        self.color = color  # Base RGB color
        self.ambient = ambient
        self.diffuse = diffuse
        self.specular = specular
        self.shininess = shininess


class Light:
    def __init__(self, position, intensity=(1, 1, 1)):
        self.position = position
        self.intensity = intensity  # RGB tuple


import numpy as np


def normalize(v):
    norm = np.linalg.norm(v)
    return v / norm if norm > 0 else v


def phong_shading(point, normal, view_dir, material, light):
    normal = normalize(normal)
    light_dir = normalize(np.array(light.position) - np.array(point))
    reflect_dir = normalize(2 * normal * np.dot(normal, light_dir) - light_dir)

    ambient = np.array(material.color) * material.ambient

    diff = max(np.dot(normal, light_dir), 0.0)
    diffuse = np.array(material.color) * material.diffuse * diff


    spec = max(np.dot(view_dir, reflect_dir), 0.0) ** material.shininess
    specular = np.array(light.intensity) * material.specular * spec

    color = ambient + diffuse + specular
    color = np.clip(color, 0, 1) * 255

    return tuple(color.astype(int))


def transform_scene(scene, matrix):
    scene.vertices = apply_transformations(scene.vertices, matrix)


def apply_transformations(vertices, matrix):
    transformed = []
    for v in vertices:
        v_homogeneous = np.array([v[0], v[1], v[2], 1])  # A 3D point extended to 4D to allow translations
        v_transformed = matrix @ v_homogeneous  # Matrix multiplication
        transformed.append(v_transformed[:3])
    return transformed


def translation_matrix(tx, ty, tz):
    return np.array([
        [1, 0, 0, tx],
        [0, 1, 0, ty],
        [0, 0, 1, tz],
        [0, 0, 0, 1]
    ])


def scaling_matrix(sx, sy, sz):
    return np.array([
        [sx, 0, 0, 0],
        [0, sy, 0, 0],
        [0, 0, sz, 0],
        [0, 0, 0, 1]
    ])


def rotation_matrix_x(theta):
    c, s = np.cos(theta), np.sin(theta)
    return np.array([
        [1, 0, 0, 0],
        [0, c, -s, 0],
        [0, s, c, 0],
        [0, 0, 0, 1]
    ])


def rotation_matrix_y(theta):
    c, s = np.cos(theta), np.sin(theta)
    return np.array([
        [c, 0, s, 0],
        [0, 1, 0, 0],
        [-s, 0, c, 0],
        [0, 0, 0, 1]
    ])


def rotation_matrix_z(theta):
    c, s = np.cos(theta), np.sin(theta)
    return np.array([
        [c, -s, 0, 0],
        [s, c, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1]
    ])


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
    # vectors
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

import math


def project(v, width, height, fov):
    scale = width / (2 * math.tan(math.radians(fov / 2)))
    x = v[0] * scale / v[2] + width / 2
    y = -v[1] * scale / v[2] + height / 2
    return (x, y, v[2])


# Extract Triangles from the OBJ File
def load_triangles_from_obj(scene, width, height, camera, material):
    triangles = []
    vertices = scene.vertices

    for mesh in scene.mesh_list:
        for face in mesh.faces:
            if len(face) != 3:
                continue

            v1 = camera.world_to_camera(vertices[face[0]])
            v2 = camera.world_to_camera(vertices[face[1]])
            v3 = camera.world_to_camera(vertices[face[2]])

            # Skip triangles behind the camera
            if v1[2] <= 0 or v2[2] <= 0 or v3[2] <= 0:
                continue

            p1 = project(v1, width, height, camera.fov)
            p2 = project(v2, width, height, camera.fov)
            p3 = project(v3, width, height, camera.fov)

            normal = normalize(np.cross(np.subtract(v2, v1), np.subtract(v3, v1)))
            view_dir = normalize(camera.position - v1)

            # Use transformed light for shading
            color = phong_shading(v1, normal, view_dir, material, light_world)
            triangles.append((p1, p2, p3, color))

    return triangles


import copy

#light_world = Light(position=(0, -500, 500), intensity=(255, 255, 255))
light_world = Light(position=(0, 0, 10), intensity=(1.0, 1.0, 1.0))
# dragon scene
scene = pywavefront.Wavefront('dragon/Untitled.obj', collect_faces=True)
#
#
# scene_original = pywavefront.Wavefront('box/box.obj', collect_faces=True)
# original_vertices = copy.deepcopy(scene_original.vertices)
#
camera = Camera(
    position=(0, 5, 5),        # Camera is directly in front
    look_at=(0, 0, 1),         # Looking at center of the scene
    up=[0, 1, 0],             # Y-axis points downward on screen
    fov=90
)
## Video creation
# width, height = 1000, 1000
# frames = 60
#
# for i in range(frames):
#     t = i / frames
#
#     # Rotation angles in degrees
#     angle_x = 30 * np.sin(2 * np.pi * t)
#     angle_y = 360 * t
#     angle_z = 20 * np.cos(2 * np.pi * t)
#
#     # Translation and scale oscillation
#     tx = 0.5 * np.sin(2 * np.pi * t)
#     ty = 0.2 * np.cos(2 * np.pi * t)
#     scale = 1 + 0.1 * np.sin(2 * np.pi * t)
#
#     # Build transformations
#     Rx = rotation_matrix_x(angle_x)
#     Ry = rotation_matrix_y(angle_y)
#     Rz = rotation_matrix_z(angle_z)
#     T = translation_matrix(tx, ty, 0)
#     S = scaling_matrix(scale, scale, scale)
#
#     # Combined transformation: T * Rz * Ry * Rx * S
#     transformation = T @ Rz @ Ry @ Rx @ S
#
#     # Apply to fresh scene
#     scene_frame = pywavefront.Wavefront('box/box.obj', collect_faces=True)
#     scene_frame.vertices = copy.deepcopy(original_vertices)
#     transform_scene(scene_frame, transformation)
#
#     # Render frame
#     image = Image.new("RGB", (width, height), (255, 255, 255))
#     zbuffer = [[float('inf')] * width for _ in range(height)]
#
#     triangles = load_triangles_from_obj(scene_frame, width, height, camera)
#
#     for t in triangles:
#         p1, p2, p3 = t[0], t[1], t[2]
#         color = t[3]
#         rasterize_triangle(image, zbuffer, p1, p2, p3, color)
#
#     image.save(f"frames/frame_{i:03d}.png")

# Diffuse (green) object on the left
# material_diffuse = Material(
#     color=(0, 1.0, 0),
#     ambient=0.1,
#     diffuse=1.0,
#     specular=0.0,
#     shininess=0
# )

# Specular (blue) object on the right
material_specular = Material(
    color=(0, 1.0, 0),
    ambient=0.1,
    diffuse=0.3,
    specular=1,
    shininess=2
)
# # Load two box objects
# scene1 = pywavefront.Wavefront('box/box.obj', collect_faces=True)
# scene2 = pywavefront.Wavefront('box/box.obj', collect_faces=True)

# Translate left and right, also raise up so they're on-screen
# transform_scene(scene1, translation_matrix(-2, 0, 0))  # Left, upward
# transform_scene(scene2, translation_matrix(2, 0, 0))   # Right, upward

# Convert objects into triangles
# triangles1 = load_triangles_from_obj(scene1, width, height, camera, material_diffuse)
# triangles2 = load_triangles_from_obj(scene2, width, height, camera, material_specular)
# triangles = triangles1 + triangles2
triangles = load_triangles_from_obj(scene, width, height, camera, material_specular)

# rasterize triangles, devide their x and y by z, and also, move the (0,0) to the center of the picture using width/2 for x and height / 2 for y like shown below.
for t in triangles:
    p1, p2, p3 = t[0], t[1], t[2]
    color = t[3]
    rasterize_triangle(image, zbuffer, p1, p2, p3, color)
image.save("img.png")
image.show()
