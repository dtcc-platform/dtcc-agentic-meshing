from typing import List, Tuple, Optional, Dict
import time 
import re
import meshio
import numpy as np

def tic():
    global _t
    _t = time.time()


def toc(task: Optional[str] = None):
    global _t
    if _t is None:
        raise RuntimeError("Must call tic() before toc()")
    dt = time.time() - _t
    _t = None
    if task:
        print(f"Elapsed time: {dt:.3f} s for {task}")
    else:
        print(f"Elapsed time: {dt:.3f} s")
    return dt


# ------------- Types -------------
XY = Tuple[float, float]
Loop = List[XY]

def signed_area(poly: Loop) -> float:
    """Positive for CCW, negative for CW."""
    A = 0.0
    for (x1, y1), (x2, y2) in zip(poly, poly[1:] + poly[:1]):
        A += x1 * y2 - x2 * y1
    return 0.5 * A


def polygon_centroid(poly: Loop) -> XY:
    """Centroid for simple polygon; falls back to vertex average if degenerate."""
    A2 = 0.0
    Cx = 0.0
    Cy = 0.0
    for (x1, y1), (x2, y2) in zip(poly, poly[1:] + poly[:1]):
        c = x1 * y2 - x2 * y1
        A2 += c
        Cx += (x1 + x2) * c
        Cy += (y1 + y2) * c
    if abs(A2) < 1e-30:
        xs, ys = zip(*poly)
        n = len(poly)
        return (sum(xs) / n, sum(ys) / n)
    A = A2 / 2.0
    return (Cx / (6 * A), Cy / (6 * A))


def point_in_polygon(pt: XY, poly: Loop) -> bool:
    """Ray casting; True if inside (treats boundary as outside for robustness)."""
    x, y = pt
    inside = False
    for (x1, y1), (x2, y2) in zip(poly, poly[1:] + poly[:1]):
        if (y1 > y) != (y2 > y):
            xin = (x2 - x1) * (y - y1) / (y2 - y1 + 1e-300) + x1
            if xin > x:
                inside = not inside
    return inside


def tri_centroid(points_xyz: np.ndarray, tri: np.ndarray) -> XY:
    a, b, c = points_xyz[tri[0]], points_xyz[tri[1]], points_xyz[tri[2]]
    return ((a[0] + b[0] + c[0]) / 3.0, (a[1] + b[1] + c[1]) / 3.0)


# ------------- File I/O helpers -------------


def tokens_to_pairs(tokens: List[str]) -> Loop:
    if len(tokens) % 2 != 0:
        raise ValueError(
            f"Odd number of coordinates ({len(tokens)}); expected x y x y ..."
        )
    it = iter(tokens)
    return [(float(x), float(y)) for x, y in zip(it, it)]


def dedupe_closing_vertex(coords: Loop, tol: float = 1e-12) -> Loop:
    """Remove last point if it duplicates the first."""
    if len(coords) >= 2:
        x0, y0 = coords[0]
        x1, y1 = coords[-1]
        if abs(x0 - x1) <= tol and abs(y0 - y1) <= tol:
            return coords[:-1]
    return coords


def read_loops_from_file(path: str) -> Tuple[Loop, List[Loop]]:
    """
    Format:
      - First non-empty line: outer boundary "x1 y1 x2 y2 ..."
      - Subsequent non-empty lines: additional inner loops (holes/islands) in same format
    Returns: (exterior, [inner_loops...])
    """
    tic()
    lines: List[str] = []
    with open(path, "r") as f:
        for raw in f:
            line = raw.strip()
            if line:
                lines.append(line)
    if not lines:
        raise ValueError("No non-empty lines found")

    def parse_line(line: str) -> Loop:
        tokens = re.split(r"\s+", line.strip())
        pts = dedupe_closing_vertex(tokens_to_pairs(tokens))
        if len(pts) < 3:
            raise ValueError("Polygon has < 3 vertices")
        return pts

    exterior = parse_line(lines[0])
    inner_loops = [parse_line(ln) for ln in lines[1:]]

    # Basic sanity: ensure each inner loop lies within exterior by centroid
    for idx, loop in enumerate(inner_loops):
        c = polygon_centroid(loop)
        if not point_in_polygon(c, exterior):
            raise ValueError(f"Inner loop #{idx+1} centroid not inside exterior")
    toc(f"read {len(lines)} loops from {path}")
    return exterior, inner_loops


# ------------- Line overlay helpers (for ParaView) -------------


def build_point_lookup(
    points_xyz: np.ndarray, ndp: int = 12
) -> Dict[Tuple[float, float], int]:
    """Map rounded (x,y) -> point index."""
    lut = {}
    for i, (x, y, _) in enumerate(points_xyz):
        lut[(round(x, ndp), round(y, ndp))] = i
    return lut


def map_loop_to_node_ids(
    loop: Loop,
    points_xyz: np.ndarray,
    lut: Dict[Tuple[float, float], int],
    ndp: int = 12,
) -> List[int]:
    """Map exact loop vertices to nearest mesh vertices (fallback to nearest if rounding misses)."""
    ids: List[int] = []
    P2 = points_xyz[:, :2]
    for x, y in loop:
        key = (round(x, ndp), round(y, ndp))
        if key in lut:
            ids.append(lut[key])
        else:
            d2 = np.sum((P2 - np.array([x, y])) ** 2, axis=1)
            ids.append(int(np.argmin(d2)))
    return ids


# ------------- Nesting/metadata helpers -------------


def sample_point_inside(poly: Loop, eps: float = 1e-3) -> XY:
    """
    Return a point guaranteed to be inside 'poly' but biased toward a vertex,
    so it cannot lie inside an inner loop of 'poly' (for non-overlapping rings).
    """
    c = polygon_centroid(poly)
    v0x, v0y = poly[0]
    return (c[0] + (v0x - c[0]) * (1.0 - eps), c[1] + (v0y - c[1]) * (1.0 - eps))


def compute_loop_metadata(
    exterior: Loop, inner_loops: Optional[List[Loop]]
) -> List[Dict]:
    """
    Build list of loops with:
      poly, ccw, sample_pt (inside loop), depth (containers count)
    Depth is computed by counting how many OTHER loops contain the loop's *sample point*
    (near a vertex), which avoids misclassifying the outer loop when its centroid
    lies inside an inner ring.
    """
    loops: List[Dict] = [{"name": "outer", "poly": exterior}]
    for k, lp in enumerate(inner_loops or []):
        loops.append({"name": f"loop{k}", "poly": lp})

    for L in loops:
        L["ccw"] = signed_area(L["poly"]) > 0
        L["sample_pt"] = sample_point_inside(L["poly"])  # inside this loop

    for i, Li in enumerate(loops):
        depth = 0
        pt = Li["sample_pt"]
        for j, Lj in enumerate(loops):
            if j == i:
                continue
            if point_in_polygon(pt, Lj["poly"]):
                depth += 1
        Li["depth"] = depth
    return loops


# ------------- VTU writers -------------


def save_vtu(
    points_xyz: np.ndarray,
    triangles: np.ndarray,
    edges: Optional[np.ndarray] = None,
    filename: str = "mesh.vtu",
) -> None:
    tic()
    cells = [("triangle", triangles)]
    if edges is not None and len(edges) > 0:
        cells.append(("line", edges))
    meshio.Mesh(points=points_xyz, cells=cells).write(filename)
    print(
        f"Saved {filename} (triangles={len(triangles)}, lines={0 if edges is None else len(edges)})"
    )
    toc(f"write {filename}")


def save_vtu_with_regions_and_lines(
    points_xyz: np.ndarray,
    triangles: np.ndarray,
    regions: np.ndarray,
    line_cells: Optional[np.ndarray] = None,
    filename: str = "mesh_regions.vtu",
) -> None:
    tic()
    cells = [("triangle", triangles)]
    cell_data = {"region": [regions]}
    if line_cells is not None and len(line_cells) > 0:
        cells.append(("line", line_cells))
        cell_data["region"].append(
            np.zeros(len(line_cells), dtype=np.int32)
        )  # pad for line block
    meshio.Mesh(points=points_xyz, cells=cells, cell_data=cell_data).write(filename)
    print(
        f"Saved {filename} (triangles={len(triangles)}, lines={0 if line_cells is None else len(line_cells)})"
    )
    toc(f"write {filename}")
