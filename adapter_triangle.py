from __future__ import annotations

from typing import List, Tuple, Optional

import numpy as np
import triangle as tr

XY = Tuple[float, float]
Loop = List[XY]


def _loop_segments(points: Loop, start_index: int) -> List[Tuple[int, int]]:
    segs: List[Tuple[int, int]] = []
    count = len(points)
    for i in range(count):
        a = start_index + i
        b = start_index + ((i + 1) % count)
        segs.append((a, b))
    return segs


def _polygon_centroid(loop: Loop) -> Tuple[float, float]:
    area_acc = 0.0
    cx_acc = 0.0
    cy_acc = 0.0
    for (x0, y0), (x1, y1) in zip(loop, loop[1:] + loop[:1]):
        cross = x0 * y1 - x1 * y0
        area_acc += cross
        cx_acc += (x0 + x1) * cross
        cy_acc += (y0 + y1) * cross
    if area_acc == 0.0:
        xs = [p[0] for p in loop]
        ys = [p[1] for p in loop]
        return (sum(xs) / len(xs), sum(ys) / len(ys))
    area = area_acc / 2.0
    return (cx_acc / (6.0 * area), cy_acc / (6.0 * area))


def triangulate(
    outer: Loop,
    inner_loops: List[Loop],
    *,
    maxh: Optional[float],
    quality: str,
    enforce_constraints: bool,
):
    vertices: List[Tuple[float, float]] = []
    segments: List[Tuple[int, int]] = []

    vertices.extend(outer)
    offset = len(outer)

    holes: List[Tuple[float, float]] = []
    for loop in inner_loops:
        vertices.extend(loop)
        segments.extend(_loop_segments(loop, offset))
        holes.append(_polygon_centroid(loop))
        offset += len(loop)

    use_pslg = enforce_constraints or bool(inner_loops)
    if use_pslg:
        segments.extend(_loop_segments(outer, 0))

    mesh: dict = {"vertices": np.asarray(vertices, dtype=np.float64)}
    if use_pslg:
        mesh["segments"] = np.asarray(segments, dtype=np.int32)
    if holes:
        mesh["holes"] = np.asarray(holes, dtype=np.float64)

    opts = "p" if use_pslg else ""
    if quality == "moderate":
        opts += "q28"
    elif quality == "default":
        pass
    else:
        pass

    if maxh is not None:
        target_area = 0.433 * float(maxh) * float(maxh)
        opts += f"a{target_area}"

    result = tr.triangulate(mesh, opts)

    pts = result.get("vertices")
    tris = result.get("triangles")
    segs = result.get("segments")
    if pts is None or tris is None:
        raise RuntimeError("Triangle returned incomplete result")

    points_xyz = [(float(x), float(y), 0.0) for x, y in pts]
    triangles = [tuple(int(idx) for idx in tri) for tri in tris]
    lines = [tuple(int(idx) for idx in seg) for seg in segs] if segs is not None else []

    return points_xyz, triangles, lines
