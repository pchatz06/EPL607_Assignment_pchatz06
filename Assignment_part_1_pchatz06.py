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


def is_inside_triangle(px, py, p1, p2, p3):
    # the [0] is the x and [1] is the y of the points
    # pc is the checked point, and p1, p2 represents the points forming and edge
    def sign(pc, p1, p2):
        return (pc[0] - p2[0]) * (p1[1] - p2[1]) - (p1[0] - p2[0]) * (pc[1] - p2[1])

    # sign of the point to each edge of the triangle (3 edges)
    e1 = sign((px, py), p1, p2)
    e2 = sign((px, py), p2, p3)
    e3 = sign((px, py), p3, p1)

    has_neg = (e1 <= 0) or (e2 <= 0) or (e3 <= 0)
    has_pos = (e1 >= 0) or (e2 >= 0) or (e3 >= 0)

    # basically here we check if the point has both positive and negative signs, it is outside.
    # to be inside, as we mentioned before it has to have 3 same signs (3 positive or 3 negative).
    return not (has_neg and has_pos)


def rasterize_triangle(image, v1, v2, v3, color):
    draw = ImageDraw.Draw(image)

    # iterate over the image and fill pixels inside triangle
    for x in range(0, image.width + 1):
        for y in range(0, image.height + 1):
            # use the function above, if it returns true, then the point in inside triangle, and color it, else its not.
            if is_inside_triangle(x, y, v1, v2, v3):
                draw.point((x, y), fill=color)
                # draw.point((x, y), fill=(random.randint(0, 255),random.randint(0, 255),random.randint(0, 255))) # Random color of each pixel


# initialize image (size and create the object as a blank white image)
width, height = 200, 200
image = Image.new("RGB", (width, height), (255, 255, 255))

# triangle points, as I have found the point (0, 0) in PIL is in the top left
p1, p2, p3 = (50, 150), (150, 150), (100, 50)
color = (0, 0, 255)
# inputs for RASTERIZE are image, points and the color of the triangle
rasterize_triangle(image, p1, p2, p3, color)
image.save("img.png")
image.show()
