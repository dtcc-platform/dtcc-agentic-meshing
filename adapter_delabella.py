import math
import subprocess
import sys
import tempfile
from pathlib import Path

_CLI = Path(__file__).resolve().parent / "delabella_cli"
_WARNED = set()

_EPS = 1e-9


def _warn_once(message: str) -> None:
    if message not in _WARNED:
        _WARNED.add(message)
        print(message, file=sys.stderr)


def _point_key(pt):
    return (round(float(pt[0]), 12), round(float(pt[1]), 12))


def _point_on_segment(pt, a, b, tol=_EPS):
    ax, ay = a
    bx, by = b
    px, py = pt
    cross = (bx - ax) * (py - ay) - (by - ay) * (px - ax)
    if abs(cross) > tol:
        return False
    dot = (px - ax) * (bx - ax) + (py - ay) * (by - ay)
    if dot < -tol:
        return False
    sq_len = (bx - ax) * (bx - ax) + (by - ay) * (by - ay)
    if dot > sq_len + tol:
        return False
    return True


def _point_in_ring(pt, ring):
    if len(ring) < 3:
        return False
    x, y = pt
    inside = False
    for i in range(len(ring)):
        x0, y0 = ring[i]
        x1, y1 = ring[(i + 1) % len(ring)]
        if _point_on_segment(pt, (x0, y0), (x1, y1)):
            return True
        if abs(y1 - y0) < _EPS:
            continue
        cond0 = y0 > y
        cond1 = y1 > y
        if cond0 == cond1:
            continue
        t = (y - y0) / (y1 - y0)
        if t < -_EPS or t > 1 + _EPS:
            continue
        xi = x0 + t * (x1 - x0)
        if abs(xi - x) <= _EPS:
            return True
        if xi > x + _EPS:
            inside = not inside
    return inside


def _point_in_polygon(pt, outer, inners):
    if not _point_in_ring(pt, outer):
        return False
    for ring in inners:
        if _point_in_ring(pt, ring):
            return False
    return True


def _add_loop(loop, add_point, edges, target_edge, include_edges):
    n = len(loop)
    for i in range(n):
        p0 = loop[i]
        p1 = loop[(i + 1) % n]
        idx0 = add_point(p0)
        idx1 = add_point(p1)
        steps = 1
        if target_edge is not None and target_edge > _EPS:
            dist = math.hypot(p1[0] - p0[0], p1[1] - p0[1])
            if dist > target_edge:
                steps = max(1, int(math.ceil(dist / target_edge)))
        mids = []
        if steps > 1:
            for step in range(1, steps):
                t = step / steps
                x = p0[0] + t * (p1[0] - p0[0])
                y = p0[1] + t * (p1[1] - p0[1])
                mid_idx = add_point((x, y))
                mids.append(mid_idx)
        if include_edges:
            last = idx0
            for mid in mids:
                edges.append((last, mid))
                last = mid
            edges.append((last, idx1))


def triangulate(outer, inner_loops, *, maxh, quality, enforce_constraints):
    if not _CLI.exists():
        raise RuntimeError("delabella_cli executable not found; build step missing")

    if quality not in {"default", "moderate"}:
        _warn_once("FeatureMissing: DelaBella adapter ignores quality option")
    elif quality == "moderate":
        _warn_once("FeatureMissing: DelaBella adapter ignores quality option")

    if maxh is None or maxh <= 0:
        target_edge = None
    else:
        target_edge = float(maxh)

    outer_loop = [(float(x), float(y)) for x, y in outer]
    inner_loops = [[(float(x), float(y)) for x, y in loop] for loop in inner_loops]

    points = []
    edges = []
    point_map = {}

    def add_point(pt):
        key = _point_key(pt)
        idx = point_map.get(key)
        if idx is not None:
            return idx
        idx = len(points)
        points.append((float(pt[0]), float(pt[1])))
        point_map[key] = idx
        return idx

    include_edges = bool(enforce_constraints)
    _add_loop(outer_loop, add_point, edges, target_edge, include_edges)
    for loop in inner_loops:
        _add_loop(loop, add_point, edges, target_edge, include_edges)

    if target_edge is not None:
        min_x = min(p[0] for p in outer_loop)
        max_x = max(p[0] for p in outer_loop)
        min_y = min(p[1] for p in outer_loop)
        max_y = max(p[1] for p in outer_loop)
        dx = target_edge
        dy = target_edge * math.sqrt(3.0) / 2.0
        if dy <= _EPS:
            dy = target_edge
        row = 0
        y = min_y - dy
        expand = target_edge
        max_x += expand
        min_x -= expand
        max_y += expand
        min_y -= expand
        while y <= max_y + _EPS:
            offset = 0.0 if row % 2 == 0 else dx * 0.5
            x = min_x - dx
            while x <= max_x + _EPS:
                pt = (x + offset, y)
                if _point_in_polygon(pt, outer_loop, inner_loops):
                    add_point(pt)
                x += dx
            y += dy
            row += 1

    if len(points) < 3:
        raise RuntimeError("Not enough points for triangulation")

    enforce_flag = 1 if (include_edges and edges) else 0

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        input_path = tmp_path / "in.txt"
        output_path = tmp_path / "out.txt"
        with input_path.open("w") as f:
            f.write(f"{len(points)} {enforce_flag}\n")
            for x, y in points:
                f.write(f"{x:.12f} {y:.12f}\n")
            f.write(f"{len(edges)}\n")
            for a, b in edges:
                f.write(f"{a} {b}\n")
        result = subprocess.run(
            [str(_CLI), str(input_path), str(output_path)],
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            err = result.stderr.strip()
            out = result.stdout.strip()
            details = "; ".join([msg for msg in (err, out) if msg])
            raise RuntimeError(f"delabella_cli failed: {details}")
        with output_path.open("r") as f:
            lines = [line.strip() for line in f if line.strip()]
        idx = 0
        vertex_count = int(lines[idx]); idx += 1
        out_points = []
        for _ in range(vertex_count):
            x_str, y_str = lines[idx].split()
            idx += 1
            out_points.append((float(x_str), float(y_str), 0.0))
        tri_count = int(lines[idx]); idx += 1
        out_tris = []
        for _ in range(tri_count):
            a_str, b_str, c_str = lines[idx].split()
            idx += 1
            out_tris.append((int(a_str), int(b_str), int(c_str)))
        seg_count = int(lines[idx]); idx += 1
        out_lines = []
        for _ in range(seg_count):
            a_str, b_str = lines[idx].split()
            idx += 1
            out_lines.append((int(a_str), int(b_str)))

    if maxh is not None and target_edge is None:
        _warn_once("FeatureMissing: Ignoring non-positive maxh request")

    return out_points, out_tris, out_lines
