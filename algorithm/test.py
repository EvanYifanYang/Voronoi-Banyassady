from classical_algorithm.foronoi import Polygon
from deterministic_algorithm import DeterministicAlgorithm

# Define a set of points
points = [
    (1, 1),
    (20, 1),
    (15, 20),
    (13, 10),
    (12, 5),
    (5, 12),
    (-5, 20),
    (19, 16),
    (10, 12),
    (9, 10)
]

# Define the batch size s
batch_size = 3

polygon = Polygon([
    (-10, -10),
    (-10, 30),
    (30, 30),
    (30, -10)
])

# Initialize the algorithm
v = DeterministicAlgorithm(polygon, batch_size)

# Create the diagram
v.algorithm_process(points=points, vis_steps=True, vis_result=True)
