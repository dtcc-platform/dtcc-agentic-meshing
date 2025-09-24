from __future__ import annotations
import math
from functools import lru_cache
from typing import List, Tuple, Optional

import numpy as np
import triangle

XY = Tuple[float, float]
XYZ = Tuple[float, float, float]
Tri = Tuple[int, int, int]
Seg = Tuple[int, int]

_SQRT3_OVER4 = math.sqrt(3.0) / 4.0


def _dedupe_loop(loop: List[XY], tol: float = 1e-12) -> List[XY]:
    if len(loop) >= 2:
        x0, y0 = loop[0]
        x1, y1 = loop[-1]
        if abs(x0 - x1) <= tol and abs(y0 - y1) <= tol:
            return loop[:-1]
    return loop


def _polygon_signed_area(loop: List[XY]) -> float:
    area = 0.0
    n = len(loop)
    for i in range(n):
        x0, y0 = loop[i]
        x1, y1 = loop[(i + 1) % n]
        area += x0 * y1 - x1 * y0
    return 0.5 * area


def _point_in_polygon(pt: XY, loop: List[XY]) -> bool:
    x, y = pt
    inside = False
    n = len(loop)
    for i in range(n):
        x0, y0 = loop[i]
        x1, y1 = loop[(i + 1) % n]
        cond = ((y0 > y) != (y1 > y)) and (x < (x1 - x0) * (y - y0) / ((y1 - y0) or 1e-30) + x0)
        if cond:
            inside = not inside
    return inside


def _polygon_centroid(loop: List[XY]) -> Tuple[float, float, float]:
    area = 0.0
    cx = 0.0
    cy = 0.0
    n = len(loop)
    for i in range(n):
        x0, y0 = loop[i]
        x1, y1 = loop[(i + 1) % n]
        cross = x0 * y1 - x1 * y0
        area += cross
        cx += (x0 + x1) * cross
        cy += (y0 + y1) * cross
    area *= 0.5
    if abs(area) < 1e-12:
        cx = sum(p[0] for p in loop) / n
        cy = sum(p[1] for p in loop) / n
        return cx, cy, area
    cx /= (3.0 * area)
    cy /= (3.0 * area)
    return cx, cy, area


def _interior_point(loop: List[XY]) -> XY:
    cx, cy, _ = _polygon_centroid(loop)
    if _point_in_polygon((cx, cy), loop):
        return (cx, cy)
    sx = sum(p[0] for p in loop) / len(loop)
    sy = sum(p[1] for p in loop) / len(loop)
    if _point_in_polygon((sx, sy), loop):
        return (sx, sy)
    x0, y0 = loop[0]
    x1, y1 = loop[1 % len(loop)]
    mx, my = (0.5 * (x0 + x1), 0.5 * (y0 + y1))
    if _point_in_polygon((mx, my), loop):
        return (mx, my)
    return (cx, cy)


def _ensure_orientation(loop: List[XY], ccw: bool) -> List[XY]:
    area = _polygon_signed_area(loop)
    if ccw and area < 0.0:
        return list(reversed(loop))
    if not ccw and area > 0.0:
        return list(reversed(loop))
    return loop


def _loop_key(loop: List[XY]) -> Tuple[XY, ...]:
    cleaned = _dedupe_loop([(float(x), float(y)) for x, y in loop])
    return tuple(cleaned)


def _classify_loops(outer: List[XY], inner_loops: List[List[XY]]):
    data = []
    outer_clean = _dedupe_loop(list(outer))
    outer_point = _interior_point(outer_clean)
    data.append(
        {
            "coords": outer_clean,
            "depth": 0,
            "is_hole": False,
            "point": outer_point,
        }
    )
    all_loops = data + []
    for loop in inner_loops:
        clean = _dedupe_loop(list(loop))
        point = _interior_point(clean)
        all_loops.append({"coords": clean, "point": point})
    for i in range(1, len(all_loops)):
        item = all_loops[i]
        depth = 0
        for j in range(len(all_loops)):
            if i == j:
                continue
            if _point_in_polygon(item["point"], all_loops[j]["coords"]):
                depth += 1
        item["depth"] = depth
        item["is_hole"] = (depth % 2 == 1)
    return all_loops


@lru_cache(maxsize=16)
def _prepare_pslg_cached(
    outer_key: Tuple[XY, ...], inner_keys: Tuple[Tuple[XY, ...], ...]
):
    outer_loop = [tuple(pt) for pt in outer_key]
    inner_loops = [[tuple(pt) for pt in loop] for loop in inner_keys]
    loops = _classify_loops(outer_loop, inner_loops)

    vertices: List[XY] = []
    segments: List[Seg] = []
    holes: List[XY] = []
    region_points: List[XY] = []
    region_markers: List[float] = []

    def add_loop(loop_coords: List[XY]) -> None:
        start = len(vertices)
        vertices.extend(loop_coords)
        n = len(loop_coords)
        for i in range(n):
            segments.append((start + i, start + (i + 1) % n))

    region_id = 1.0

    for idx, item in enumerate(loops):
        is_outer = idx == 0
        loop_coords = item["coords"]
        if is_outer:
            loop_coords = _ensure_orientation(loop_coords, ccw=True)
        elif item["is_hole"]:
            loop_coords = _ensure_orientation(loop_coords, ccw=False)
            holes.append(item["point"])
        else:
            loop_coords = _ensure_orientation(loop_coords, ccw=True)
        loops[idx]["coords"] = loop_coords
        add_loop(loop_coords)

        if is_outer or not item["is_hole"]:
            region_points.append(item["point"])
            region_markers.append(region_id)
            region_id += 1.0

    vertices_arr = np.asarray(vertices, dtype=np.float64)
    segments_arr = np.asarray(segments, dtype=np.int32)
    holes_arr = np.asarray(holes, dtype=np.float64) if holes else None
    region_pts_arr = np.asarray(region_points, dtype=np.float64)
    region_markers_arr = np.asarray(region_markers, dtype=np.float64)

    return vertices_arr, segments_arr, holes_arr, region_pts_arr, region_markers_arr


def triangulate(
    outer: List[XY],
    inner_loops: List[List[XY]],
    *,
    maxh: Optional[float],
    quality: str,
    enforce_constraints: bool,
):
    outer_key = _loop_key(outer)
    inner_key = tuple(_loop_key(loop) for loop in inner_loops)
    vertices_arr, segments_arr, holes_arr, region_pts_arr, region_markers_arr = _prepare_pslg_cached(
        outer_key, inner_key
    )

    if vertices_arr.size == 0 or segments_arr.size == 0:
        raise RuntimeError("Empty input for triangulation")

    max_area = None
    if maxh is not None and maxh > 0.0:
        max_area = _SQRT3_OVER4 * (maxh ** 2)

    mesh_in = {
        "vertices": vertices_arr,
        "segments": segments_arr,
    }
    if holes_arr is not None and holes_arr.size:
        mesh_in["holes"] = holes_arr
    if region_pts_arr.size:
        area_column = (
            np.full((region_pts_arr.shape[0], 1), max_area, dtype=np.float64)
            if max_area is not None
            else np.zeros((region_pts_arr.shape[0], 1), dtype=np.float64)
        )
        regions = np.hstack((region_pts_arr, region_markers_arr[:, None], area_column))
        mesh_in["regions"] = regions

    options = "pzQ"
    if quality == "moderate":
        options += "q28"
    if max_area is not None:
        options += f"a{max_area:.12g}"

    result = triangle.triangulate(mesh_in, options)

    if "triangles" not in result or result["triangles"] is None:
        raise RuntimeError("triangle returned no triangles")

    points_arr = result.get("vertices")
    if points_arr is None:
        raise RuntimeError("triangle returned no vertices")

    points_arr = np.asarray(points_arr, dtype=np.float64)
    tris_arr = np.asarray(result["triangles"], dtype=np.int32)
    segs_arr = result.get("segments")

    tri_pts = points_arr[tris_arr]
    v0 = tri_pts[:, 1, :] - tri_pts[:, 0, :]
    v1 = tri_pts[:, 2, :] - tri_pts[:, 0, :]
    cross = v0[:, 0] * v1[:, 1] - v0[:, 1] * v1[:, 0]
    neg_mask = cross < 0.0
    if np.any(neg_mask):
        tmp = tris_arr[neg_mask, 1].copy()
        tris_arr[neg_mask, 1] = tris_arr[neg_mask, 2]
        tris_arr[neg_mask, 2] = tmp

    points3d = np.column_stack((points_arr, np.zeros((points_arr.shape[0], 1), dtype=np.float64)))
    points: List[XYZ] = [tuple(row) for row in points3d.tolist()]
    triangles: List[Tri] = [tuple(map(int, tri)) for tri in tris_arr.tolist()]

    lines: List[Seg] = []
    if isinstance(segs_arr, np.ndarray):
        lines = [tuple(map(int, seg)) for seg in np.asarray(segs_arr, dtype=np.int32).tolist()]

    return points, triangles, lines
