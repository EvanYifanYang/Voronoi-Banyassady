from classical_algorithm.foronoi import Voronoi, Polygon
import matplotlib.pyplot as plt
import math
import gc
from typing import List, Tuple, Sequence

class DeterministicAlgorithm:
    def __init__(self, polygon: Polygon, batch_size: int):
        self.peak_memory = 0
        self.batch_size = batch_size

        self._poly_coords: Tuple[Tuple[float, float], ...] = tuple(
            (v.x, v.y) for v in polygon.polygon_vertices
        )

        self.edges: List = []
        self.points: List[Tuple[float, float]] = []
        self.finished_points: List[Tuple[float, float]] = []
        self.upcoming_points: List[Tuple[float, float]] = []
        self.finish_flag: List[Tuple[float, float]] = []

        self.small_sites: List[Tuple[float, float]] = []
        self.big_sites: List[Tuple[float, float]] = []
        self.small_sites_index: List[int] = []
        self.big_sites_index: List[int] = []

    def algorithm_process(self, points: Sequence[Tuple[float, float]], *,
                          vis_steps=False, vis_result=False):

        self.create_diagram(points, 1)
        self.create_diagram(points, 2, vis_steps=vis_steps)
        self.create_diagram(points, 3, vis_result=vis_result)

    def _new_polygon(self) -> Polygon:
        return Polygon(self._poly_coords)

    def _build_voronoi(self, pts):
        vor = Voronoi(self._new_polygon())
        vor.create_diagram(points=pts)
        return vor

    def create_diagram(self, points, phase_number, vis_steps=False, vis_result=False):
        if phase_number == 1:
            # tracemalloc.start()
            self.points = points
            self.upcoming_points = points[self.batch_size:]
            current_set = points[:self.batch_size]
            current_ray = [1] * self.batch_size
            iteration_times = [0] * self.batch_size
            self.finish_flag = [None] * self.batch_size
            current_ray_determined_point = [(x + 1, y + 1) for x, y in current_set]

            # peak_memory_max = 0

            while all(p is not None for p in current_set):
                # print(f"\nProcessing point set: {current_set}")
                # print(f"\nCurrent_ray: {current_ray}")
                # print(f"Current_ray_determined_point: {current_ray_determined_point}")
                # print(f"Current_iteration_times: {iteration_times}")
                current_set, current_ray, current_ray_determined_point, iteration_times = self.process_small_sites(current_set, current_ray, current_ray_determined_point, points, iteration_times, phase_number)
            #     iteration_current_memory, iteration_peak_memory = tracemalloc.get_traced_memory()
            #     if iteration_peak_memory > peak_memory_max:
            #         peak_memory_max = iteration_peak_memory
            # print(f"peak_memory_max: {peak_memory_max / 1024} KB")
            # tracemalloc.stop()
            self.small_sites = self.finished_points
            self.small_sites_index = [self.points.index(p) for p in self.small_sites]
            for point in current_set:
                if point:
                    self.big_sites.append(point)
                    self.big_sites_index.append(self.points.index(point))
            # print(f"Phase 1 Success! small_sites: {self.small_sites}, big_sites: {self.big_sites}")
            # print(f"small_sites_index: {self.small_sites_index}, big_sites_index: {self.big_sites_index}\n")
        if phase_number == 2:
            self.points = points
            self.upcoming_points = points[self.batch_size:]
            current_set = points[:self.batch_size]
            current_ray = [1] * self.batch_size
            iteration_times = [0] * self.batch_size
            self.finish_flag = [None] * self.batch_size
            current_ray_determined_point = [(x + 1, y + 1) for x, y in current_set]

            while all(p is not None for p in current_set):
                # print(f"\nProcessing point set: {current_set}")
                # print(f"\nCurrent_ray: {current_ray}")
                # print(f"Current_ray_determined_point: {current_ray_determined_point}")
                # print(f"Current_iteration_times: {iteration_times}")
                current_set, current_ray, current_ray_determined_point, iteration_times = self.process_small_sites(current_set, current_ray, current_ray_determined_point, points, iteration_times, phase_number)
            # print(f"Phase 2 Success! - partial result as shown")
            # print(f"len(self.edges): {len(self.edges)}\nself.edges:{self.edges}\n")
            # if vis_steps:
            #     self.plot_final_result()
        if phase_number == 3:
            self.process_big_sites(self.big_sites, points, vis_steps)
            # print(f"Phase 3 Success! - final result as shown")
            # print(f"After Phase 3: \nlen(self.edges): {len(self.edges)}\nself.edges:{self.edges}")
            if vis_result:
                self.plot_final_result()

    def plot_partial_result(self, current_set, current_edges):
        fig, ax = plt.subplots(figsize=(8, 8))
        x_coords, y_coords = zip(*self.points)
        ax.scatter(x_coords, y_coords, color='blue', label='Points')
        polygon_x, polygon_y = zip(*self._poly_coords)
        ax.plot(list(polygon_x) + [polygon_x[0]], list(polygon_y) + [polygon_y[0]], color='black',
                label='Bounding Box')
        for e in self.edges:
            if hasattr(e, 'origin') and hasattr(e, 'next'):
                start = e.origin.point
                end = e.next.origin.point
                ax.plot([start.x, end.x], [start.y, end.y], color='red')
            else:
                start, end = e
                ax.plot([start[0], end[0]], [start[1], end[1]], color='red')
        if current_edges:
            for edge in current_edges:
                if edge is not None:
                    (start, end) = edge
                    ax.plot([start[0], end[0]], [start[1], end[1]], color='purple', linewidth=2.5)
        if current_set:
            filtered_set = [p for p in current_set if p is not None]
            if filtered_set:
                x_current, y_current = zip(*filtered_set)
            else:
                x_current, y_current = [], []
            ax.scatter(x_current, y_current, color='green', label='Current Set')
        ax.set_aspect('equal', adjustable='box')
        ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1), fancybox=True, shadow=True, ncol=3)
        plt.show()

    def plot_final_result(self):
        fig, ax = plt.subplots(figsize=(8, 8))
        x_coords, y_coords = zip(*self.points)
        ax.scatter(x_coords, y_coords, color='blue', label='Points')
        polygon_x, polygon_y = zip(*self._poly_coords)
        ax.plot(list(polygon_x) + [polygon_x[0]], list(polygon_y) + [polygon_y[0]], color='black',
                label='Bounding Box')
        for e in self.edges:
            if hasattr(e, 'origin') and hasattr(e, 'next'):
                start = e.origin
                end = e.next.origin
                ax.plot([start.x, end.x], [start.y, end.y], color='red')
            else:
                start, end = e
                ax.plot([start[0], end[0]], [start[1], end[1]], color='red')
        ax.set_aspect('equal', adjustable='box')
        ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1), fancybox=True, shadow=True, ncol=3)
        plt.show()

    def process_big_sites(self, current_set, points, vis_steps):
        v = self._build_voronoi(current_set)
        E_b = v.edges

        # Delete v memory
        # v.edges.clear()
        # v._vertices.clear()
        # v.sites.clear()
        # v.doubly_connected_edge_list.clear()
        # v.status_tree = None
        # v.bounding_poly = None
        del v
        gc.collect()

        E_b_edge_incident_point = [None] * len(E_b)
        for E_b_idx, E_b_edge in enumerate(E_b):
            incident_point = (E_b_edge.incident_point.x, E_b_edge.incident_point.y)
            E_b_edge_incident_point[E_b_idx] = incident_point
        # print(f"len(E_b): {len(E_b)} \nE_b: {E_b}")

        total_points_number = len(points)
        total_iteration_number = math.ceil(total_points_number / self.batch_size)

        for i in range(total_iteration_number):
            start_index = i * self.batch_size
            end_index = min(start_index + self.batch_size, total_points_number)
            Q = points[start_index:end_index]

            union_set = list(dict.fromkeys(current_set + Q))
            v = self._build_voronoi(union_set)

            for E_b_idx, E_b_edge in enumerate(E_b):
                if E_b_edge is None or (not isinstance(E_b_edge, tuple) and (E_b_edge.next is None or E_b_edge.next.origin is None)):
                    continue
                if hasattr(E_b_edge, 'origin') and hasattr(E_b_edge, 'next'):
                    x1 = E_b_edge.origin.x
                    y1 = E_b_edge.origin.y
                    x2 = E_b_edge.next.origin.x
                    y2 = E_b_edge.next.origin.y
                else:
                    (x1, y1), (x2, y2) = E_b_edge
                for idx, point in enumerate(current_set):
                    if E_b_edge_incident_point[E_b_idx][0] == point[0] and E_b_edge_incident_point[E_b_idx][1] == point[1]:
                        cell_edges, invalid_cell_flag = self.get_cell_edges(v, idx)
                        if invalid_cell_flag:
                            continue
                        intersection_1 = None
                        intersection_2 = None
                        if self.check_point_in_cell((x1, y1), cell_edges) and self.check_point_in_cell((x2, y2), cell_edges):
                            continue
                        for edge in cell_edges:
                            intersection, flag_continue, flag_break = self.segment_segment_intersection(edge, E_b_edge)
                            if flag_break:
                                break
                            if flag_continue:
                                continue
                            if intersection is not None:
                                if intersection_1 is None:
                                    intersection_1 = intersection
                                else:
                                    intersection_2 = intersection
                                    break
                        if intersection_1 is not None and intersection_2 is not None:
                            candidate_edge = (intersection_1, intersection_2)
                            E_b[E_b_idx] = self.check_edge_direction(point, candidate_edge)
                        elif intersection_1 is not None and intersection_2 is None:
                            if self.check_point_in_cell((x1, y1), cell_edges):
                                E_b[E_b_idx] = self.check_edge_direction(point, ((x1, y1), intersection_1))
                            else:
                                E_b[E_b_idx] = self.check_edge_direction(point, (intersection_1, (x2, y2)))
                        else:
                            E_b[E_b_idx] = None

            # Delete v memory
            # v.edges.clear()
            # v._vertices.clear()
            # v.sites.clear()
            # v.doubly_connected_edge_list.clear()
            # v.status_tree = None
            # v.bounding_poly = None
            del v
            gc.collect()

        for e in E_b:
            if e is not None:
                self.edges.append(e)

        if vis_steps:
            self.plot_partial_result(current_set, E_b)


    def process_small_sites(self, current_set, current_ray, current_determined_point, points, iteration_times, phase_number):
        iteration_times = [x + 1 for x in iteration_times]
        total_points_number = len(points)
        total_iteration_number = math.ceil(total_points_number / self.batch_size)

        # Phase 1:
        # print(f"### Phase 1: ###")
        current_line = [None] * len(current_set)
        edge_site_pair = [None] * len(current_set)
        nearest_distance = [float('inf')] * len(current_set)

        for i in range(total_iteration_number):
            # print(f"\nProcessing Iteration: {i + 1}")
            start_index = i * self.batch_size
            end_index = min(start_index + self.batch_size, total_points_number)
            Q = points[start_index:end_index]

            union_set = list(dict.fromkeys(current_set + Q))
            # print(f"union_set: {union_set}")
            # print(f"len(union_set): {len(union_set)})")
            v = self._build_voronoi(union_set)

            for idx, point in enumerate(current_set):
                cell_edges, invalid_cell_flag = self.get_cell_edges(v, idx)
                if invalid_cell_flag:
                    continue
                # print(f"idx[{idx}]: cell_edges: {cell_edges}")
                for edge in cell_edges:
                    candidate_distance = self.distance_to_intersection(point, current_ray[idx], current_determined_point[idx], edge)
                    if candidate_distance is not None and candidate_distance <= nearest_distance[idx]:
                        nearest_distance[idx] = candidate_distance
                        current_line[idx] = edge

                        index_i, index_j = self.get_site_pair_from_edge(edge, points)
                        edge_site_pair[idx] = (index_i, index_j)
            # print(f"current_line: {current_line}")
            # print(f"nearest_distance: {nearest_distance}")
            # current_memory, peak_memory = tracemalloc.get_traced_memory()
            # if peak_memory > self.peak_memory:
            #     self.peak_memory = peak_memory
            # Delete v memory
            # v.edges.clear()
            # v._vertices.clear()
            # v.sites.clear()
            # v.doubly_connected_edge_list.clear()
            # v.status_tree = None
            # v.bounding_poly = None
            del v
            gc.collect()

        # print(f"current_line: {current_line}\n")

        # print(f"### Phase 2: ###")
        # Phase 2:
        current_edges = [None] * len(current_set)
        first_edge_flag = [False] * len(current_set)
        for i in range(total_iteration_number):
            start_index = i * self.batch_size
            end_index = min(start_index + self.batch_size, total_points_number)
            Q = points[start_index:end_index]

            union_set = list(dict.fromkeys(current_set + Q))
            v = self._build_voronoi(union_set)
            # fig, ax = v.create_diagram(points=union_set, vis_result=False)

            for idx, point in enumerate(current_set):
                cell_edges, invalid_cell_flag = self.get_cell_edges(v, idx)
                if invalid_cell_flag:
                    continue
                intersection_1 = None
                intersection_2 = None
                if first_edge_flag[idx] == False:
                    # print(f"(i == 0), cell_edges: {cell_edges} for idx: {idx}, point: {point}")
                    for edge in cell_edges:
                        intersection = self.segment_line_intersection(edge, current_line[idx])
                        # print(f"edge: {edge}, intersection: {intersection}")
                        if intersection is not None:
                            if intersection_1 is None:
                                intersection_1 = intersection
                            else:
                                intersection_2 = intersection
                                break
                    if intersection_1 is not None and intersection_2 is not None:
                        candidate_edge = (intersection_1, intersection_2)
                        # print(f"(i == 0), candidate_edge: {candidate_edge} for idx: {idx}, point: {point}")
                        current_edges[idx] = self.check_edge_direction(point, candidate_edge)
                        first_edge_flag[idx] = True
                else:
                    if self.check_point_in_cell(current_edges[idx][0], cell_edges) and self.check_point_in_cell(current_edges[idx][1], cell_edges):
                        continue
                    for edge in cell_edges:
                        intersection, flag_continue, flag_break = self.segment_segment_intersection(edge, current_edges[idx])
                        if flag_break:
                            break
                        if flag_continue:
                            continue
                        if intersection is not None:
                            if intersection_1 is None:
                                intersection_1 = intersection
                            else:
                                intersection_2 = intersection
                                break
                    # print(f"intersection_1: {intersection_1}, intersection_2: {intersection_2}")
                    if intersection_1 is not None and intersection_2 is not None:
                        candidate_edge = (intersection_1, intersection_2)
                        current_edges[idx] = self.check_edge_direction(point, candidate_edge)
                    elif intersection_1 is not None and intersection_2 is None:
                        if self.check_point_in_cell(current_edges[idx][0], cell_edges):
                            current_edges[idx] = self.check_edge_direction(point, (current_edges[idx][0], intersection_1))
                        else:
                            current_edges[idx] = self.check_edge_direction(point, (intersection_1, current_edges[idx][1]))

            # current_memory, peak_memory = tracemalloc.get_traced_memory()
            # if peak_memory > self.peak_memory:
            #     self.peak_memory = peak_memory

            # Delete v memory
            # v.edges.clear()
            # v._vertices.clear()
            # v.sites.clear()
            # v.doubly_connected_edge_list.clear()
            # v.status_tree = None
            # v.bounding_poly = None
            del v
            gc.collect()

            # print(f"iteration number: {i + 1},current_edge: {current_edges}")
            # plot_edge_cut
            # self.plot_edge_cut(current_edges, fig, ax)

        # Check if iteration time = 1, store the finish current cell flag (finish point)
        for idx, iteration_time in enumerate(iteration_times):
            if iteration_time == 1 and current_edges[idx] is not None:
                self.finish_flag[idx] = current_edges[idx][0]

        # Check if finished the current cell
        for idx, current_edge in enumerate(current_edges):
            if current_edge is not None and self.finish_flag[idx] is not None and math.isclose(current_edge[1][0], self.finish_flag[idx][0], rel_tol=1e-6) and math.isclose(current_edge[1][1], self.finish_flag[idx][1], rel_tol=1e-6):
                if self.upcoming_points:
                    self.finished_points.append(current_set[idx])
                    next_point = self.upcoming_points.pop(0)
                    current_set[idx] = next_point
                    self.finish_flag[idx] = None
                    iteration_times[idx] = 0
                    current_ray[idx] = 1
                    x, y = current_set[idx]
                    current_determined_point[idx] = (x + 1, y + 1)
                else:
                    self.finished_points.append(current_set[idx])
                    self.finish_flag[idx] = None
                    current_set[idx] = None
                    iteration_times[idx] = 0
                    current_ray[idx] = 1
            else:
                # Update ray slope & current determined point
                new_ray_slope, new_determined_point = self.update_ray_slope_and_determined_point(current_set, idx, current_edge)
                current_ray[idx] = new_ray_slope
                current_determined_point[idx] = new_determined_point

        if phase_number == 2:
            for idx, edge in enumerate(current_edges):
                index_i, index_j = edge_site_pair[idx]
                if index_j is None:
                    self.edges.append(edge)
                i_small = index_i in self.small_sites_index
                j_small = index_j in self.small_sites_index
                j_big = index_j in self.big_sites_index

                if i_small and j_small:
                    if index_i < index_j:
                        self.edges.append(edge)

                if i_small and j_big:
                    self.edges.append(edge)
            self.plot_partial_result(current_set, current_edges)
        # print(f"current_edges: {current_edges}")
        # print(f"current_set: {current_set}, current_ray: {current_ray}, current_determined_point: {current_determined_point}, iteration_times: {iteration_times}")
        return current_set, current_ray, current_determined_point, iteration_times

    def get_cell_edges(self, voronoi_diagram, idx, tol=1e-6):
        matching_point = voronoi_diagram.sites[idx]
        edges = []

        start_edge = matching_point.first_edge
        current_edge = start_edge

        if (
                start_edge is None or
                start_edge.next is None or
                start_edge.origin is None or
                start_edge.next.origin is None
        ):
            return None, True

        visited_ids = set()

        while current_edge:
            edge_id = id(current_edge)
            if edge_id in visited_ids:
                break
            visited_ids.add(edge_id)

            if (
                    current_edge.origin is None or
                    current_edge.next is None or
                    current_edge.next.origin is None
            ):
                return None, True

            x1 = current_edge.origin.x
            y1 = current_edge.origin.y
            x2 = current_edge.next.origin.x
            y2 = current_edge.next.origin.y

            dx = x2 - x1
            dy = y2 - y1
            length_squared = dx * dx + dy * dy

            if length_squared > tol * tol:
                edges.append(current_edge)

            current_edge = current_edge.next

            if current_edge is start_edge:
                break

        if len(edges) < 3 or current_edge is not start_edge:
            return edges, True

        return edges, False

    def distance_to_intersection(self, point, current_ray, current_determined_point, edge):
        px, py = point
        cdp_x, cdp_y = current_determined_point

        if hasattr(edge, 'origin') and hasattr(edge, 'next'):
            sp_x = edge.origin.x
            sp_y = edge.origin.y
            ep_x = edge.next.origin.x
            ep_y = edge.next.origin.y
        else:
            (sp_x, sp_y), (ep_x, ep_y) = edge

        if ep_x != sp_x:
            edge_slope = (ep_y - sp_y) / (ep_x - sp_x)
            if current_ray == edge_slope:
                return None
            intersection_x = (sp_y - py + px * current_ray - sp_x * edge_slope) / (current_ray - edge_slope)
        else:
            intersection_x = sp_x
        intersection_y = current_ray * (intersection_x - px) + py
        dx = cdp_x - px
        dy = cdp_y - py
        if dx != 0:
            if (intersection_x - px) * dx < 0:
                return None
        else:
            if (intersection_y - py) * dy < 0:
                return None
        distance = math.sqrt((px - intersection_x) ** 2 + (py - intersection_y) ** 2 )
        return distance

    def segment_line_intersection(self, segment, line, tol=1e-6):
        if hasattr(segment, 'origin') and hasattr(segment, 'next'):
            x1 = segment.origin.x
            y1 = segment.origin.y
            x2 = segment.next.origin.x
            y2 = segment.next.origin.y
        else:
            (x1, y1), (x2, y2) = segment

        if hasattr(line, 'origin') and hasattr(line, 'next'):
            p1 = line.origin.x
            q1 = line.origin.y
            p2 = line.next.origin.x
            q2 = line.next.origin.y
        else:
            (p1, q1), (p2, q2) = line

        dx1, dy1 = x2 - x1, y2 - y1
        dx2, dy2 = p2 - p1, q2 - q1

        denominator = dx1 * dy2 - dy1 * dx2
        if abs(denominator) < tol:
            return None

        t = ((p1 - x1) * dy2 - (q1 - y1) * dx2) / denominator
        if t < 0 - tol or t > 1 + tol:
            return None

        intersection_x = x1 + t * dx1
        intersection_y = y1 + t * dy1
        return (intersection_x, intersection_y)

    def segment_segment_intersection(self, segment1, segment2, tol = 1e-6):
        if hasattr(segment1, 'origin') and hasattr(segment1, 'next'):
            x1 = segment1.origin.x
            y1 = segment1.origin.y
            x2 = segment1.next.origin.x
            y2 = segment1.next.origin.y
        else:
            (x1, y1), (x2, y2) = segment1

        if hasattr(segment2, 'origin') and hasattr(segment2, 'next'):
            p1 = segment2.origin.x
            q1 = segment2.origin.y
            p2 = segment2.next.origin.x
            q2 = segment2.next.origin.y
        else:
            (p1, q1), (p2, q2) = segment2

        # Check the Special Situation
        # cond1 = (abs((x1 - p1) * (q2 - q1) - (y1 - q1) * (p2 - p1)) < tol and
        #          min(p1, p2) - tol <= x1 <= max(p1, p2) + tol and
        #          min(q1, q2) - tol <= y1 <= max(q1, q2) + tol)
        #
        # cond2 = (abs((x2 - p1) * (q2 - q1) - (y2 - q1) * (p2 - p1)) < tol and
        #          min(p1, p2) - tol <= x2 <= max(p1, p2) + tol and
        #          min(q1, q2) - tol <= y2 <= max(q1, q2) + tol)
        #
        # flag_continue = cond1 and cond2

        cond3 = (abs((p1 - x1) * (y2 - y1) - (q1 - y1) * (x2 - x1)) < tol and
                 min(x1, x2) - tol <= p1 <= max(x1, x2) + tol and
                 min(y1, y2) - tol <= q1 <= max(y1, y2) + tol)

        cond4 = (abs((p2 - x1) * (y2 - y1) - (q2 - y1) * (x2 - x1)) < tol and
                 min(x1, x2) - tol <= p2 <= max(x1, x2) + tol and
                 min(y1, y2) - tol <= q2 <= max(y1, y2) + tol)

        flag_break = cond3 and cond4

        dx1 = x2 - x1
        dy1 = y2 - y1
        dx2 = p2 - p1
        dy2 = q2 - q1

        cond_collinear = abs((x1 - p1) * dy2 - (y1 - q1) * dx2) < tol and abs(dx1 * dy2 - dy1 * dx2) < tol
        flag_continue = cond_collinear

        denominator = dx1 * dy2 - dy1 * dx2
        if denominator == 0:
            return None, False, False

        t = ((p1 - x1) * dy2 - (q1 - y1) * dx2) / denominator
        u = ((p1 - x1) * dy1 - (q1 - y1) * dx1) / denominator

        if not (0 - tol <= t <= 1 + tol and 0 - tol <= u <= 1 + tol):
            return None, False, False

        intersection_x = x1 + t * dx1
        intersection_y = y1 + t * dy1
        return (intersection_x, intersection_y), flag_continue, flag_break

    def check_point_in_cell(self, point, edges, tol=1e-6):
        x, y = point
        ref = None

        for edge in edges:

            if hasattr(edge, 'origin') and hasattr(edge, 'next'):
                x1 = edge.origin.x
                y1 = edge.origin.y
                x2 = edge.next.origin.x
                y2 = edge.next.origin.y
            else:
                print(edge)
                (x1, y1), (x2, y2) = edge

            dx = x2 - x1
            dy = y2 - y1
            dxp = x - x1
            dyp = y - y1
            cross = dx * dyp - dy * dxp

            if abs(cross) < tol:
                continue

            if ref is None:
                ref = 1 if cross > 0 else -1
            else:
                if (cross > 0 and ref < 0) or (cross < 0 and ref > 0):
                    return False
        return True

    def plot_edge_cut(self, current_edge, fig, ax):
        for edge in current_edge:
            if edge is not None:
                (start, end) = edge
                if start is not None and end is not None:
                    ax.plot(
                        [start[0], end[0]],
                        [start[1], end[1]],
                        linestyle='--',
                        color='purple',
                        linewidth=3,
                        label='Edge Cut' if edge == current_edge[0] else None  # 只添加一次图例
                    )
        handles, labels = ax.get_legend_handles_labels()
        if 'Edge Cut' in labels:
            ax.legend(loc='upper right')
        fig.canvas.draw()
        plt.show()

    def update_ray_slope_and_determined_point(self, current_set, idx, current_edge, tol = 1e-6):
        current_point = current_set[idx]
        start_point = current_edge[0]
        end_point = current_edge[1]
        current_point_x, current_point_y = current_point
        start_point_x, start_point_y = start_point
        end_point_x, end_point_y = end_point

        temp = 0.000001

        if abs(start_point_x - end_point_x) < tol and abs(start_point_y - end_point_y) < tol:
            temp = 0.1
        if current_point_x == end_point_x:
            if end_point_y < current_point_y:
                determined_point_x = end_point_x - temp
            else:
                determined_point_x = end_point_x + temp
            determined_point_y = end_point_y
            ray_slope = (current_point_y - determined_point_y) / (current_point_x - determined_point_x)
            return ray_slope, (determined_point_x, determined_point_y)
        elif (current_point_x < end_point_x and current_point_y < end_point_y) or (current_point_x < end_point_x and current_point_y > end_point_y):
            ray_slope = (current_point_y - end_point_y) / (current_point_x - end_point_x)
            ray_slope -= temp
            determined_point_x = current_point_x + 1
            determined_point_y = ray_slope * (determined_point_x - current_point_x) + current_point_y
            return ray_slope, (determined_point_x, determined_point_y)
        elif (current_point_x > end_point_x and current_point_y > end_point_y) or (current_point_x > end_point_x and current_point_y < end_point_y):
            ray_slope = (current_point_y - end_point_y) / (current_point_x - end_point_x)
            ray_slope -= temp
            determined_point_x = current_point_x - 1
            determined_point_y = ray_slope * (determined_point_x - current_point_x) + current_point_y
            return ray_slope, (determined_point_x, determined_point_y)
        elif current_point_y == end_point_y and current_point_x < end_point_x:
            ray_slope = -temp
            determined_point_x = current_point_x + 1
            determined_point_y = ray_slope * (determined_point_x - current_point_x) + current_point_y
            return ray_slope, (determined_point_x, determined_point_y)
        elif current_point_y == end_point_y and current_point_x > end_point_x:
            ray_slope = -temp
            determined_point_x = current_point_x - 1
            determined_point_y = ray_slope * (determined_point_x - current_point_x) + current_point_y
            return ray_slope, (determined_point_x, determined_point_y)

    def check_edge_direction(self, point, edge):
        (ax, ay) = point
        (bx, by), (cx, cy) = edge

        cross = (bx - ax) * (cy - ay) - (by - ay) * (cx - ax)

        if cross < 0:
            return ((bx, by), (cx, cy))
        else:
            return ((cx, cy), (bx, by))

    def get_site_pair_from_edge(self, edge, points):
        site_i_point = edge.incident_point
        site_j_point = edge.twin.incident_point
        index_i = None
        index_j = None
        if site_i_point is not None:
            site_i_point_x, site_i_point_y = site_i_point.x, site_i_point.y
            for idx, point in enumerate(points):
                if point[0] == site_i_point_x and point[1] == site_i_point_y:
                    index_i = idx
                    break
        if site_j_point is not None:
            site_j_point_x, site_j_point_y = site_j_point.x, site_j_point.y
            for idx, point in enumerate(points):
                if point[0] == site_j_point_x and point[1] == site_j_point_y:
                    index_j = idx
                    break
        return index_i, index_j