import subprocess
import tempfile
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple

Point = Tuple[float, float]
Triangle = Tuple[int, int, int]
Segment = Tuple[int, int]

CLI_PATH = Path(__file__).resolve().parent / "cdt_triangulate"
TARGET_AREA_FACTOR = 0.4330127018922193  # approx sqrt(3)/4

_QUALITY_WARNED = False


def _build_edges(loops: Sequence[Sequence[Point]]) -> Tuple[List[Point], List[Segment]]:
    points: List[Point] = []
    edges: List[Segment] = []
    for loop in loops:
        if len(loop) < 3:
            raise ValueError("Each loop must have at least 3 vertices")
        start = len(points)
        points.extend(loop)
        n = len(loop)
        for i in range(n):
            v0 = start + i
            v1 = start + (i + 1) % n
            edges.append((v0, v1))
    return points, edges


def _write_input(path: Path, points: Sequence[Point], edges: Sequence[Segment]) -> None:
    with path.open("w", encoding="ascii") as f:
        f.write(f"{len(points)} {len(edges)}\n")
        for x, y in points:
            f.write(f"{x} {y}\n")
        for a, b in edges:
            f.write(f"{a} {b}\n")


def _parse_output(path: Path) -> Tuple[List[Tuple[float, float, float]], List[Triangle], List[Segment]]:
    with path.open("r", encoding="ascii") as f:
        header = f.readline()
        if not header:
            raise RuntimeError("cdt_triangulate produced empty output")
        parts = header.strip().split()
        if len(parts) != 3:
            raise RuntimeError(f"Unexpected header in output: {header!r}")
        n_points, n_tris, n_edges = map(int, parts)

        pts: List[Tuple[float, float, float]] = []
        for _ in range(n_points):
            line = f.readline()
            if not line:
                raise RuntimeError("Unexpected EOF while reading points")
            xs = line.strip().split()
            if len(xs) != 2:
                raise RuntimeError(f"Invalid point line: {line!r}")
            x, y = map(float, xs)
            pts.append((x, y, 0.0))

        tris: List[Triangle] = []
        for _ in range(n_tris):
            line = f.readline()
            if not line:
                raise RuntimeError("Unexpected EOF while reading triangles")
            parts = line.strip().split()
            if len(parts) != 3:
                raise RuntimeError(f"Invalid triangle line: {line!r}")
            tris.append(tuple(int(p) for p in parts))  # type: ignore[arg-type]

        edges: List[Segment] = []
        for _ in range(n_edges):
            line = f.readline()
            if not line:
                raise RuntimeError("Unexpected EOF while reading edges")
            parts = line.strip().split()
            if len(parts) != 2:
                raise RuntimeError(f"Invalid edge line: {line!r}")
            edges.append(tuple(int(p) for p in parts))  # type: ignore[arg-type]

    return pts, tris, edges


def _point_on_segment(px: float, py: float, ax: float, ay: float, bx: float, by: float, tol: float = 1e-9) -> bool:
    cross = (bx - ax) * (py - ay) - (by - ay) * (px - ax)
    if abs(cross) > tol:
        return False
    dot = (px - ax) * (px - bx) + (py - ay) * (py - by)
    return dot <= tol


def _point_in_loop(point: Tuple[float, float], loop: Sequence[Point]) -> bool:
    x, y = point
    inside = False
    n = len(loop)
    for i in range(n):
        x1, y1 = loop[i]
        x2, y2 = loop[(i + 1) % n]
        if _point_on_segment(x, y, x1, y1, x2, y2):
            return True
        intersects = ((y1 > y) != (y2 > y))
        if intersects:
            denom = y2 - y1
            if abs(denom) < 1e-18:
                continue
            x_int = x1 + (y - y1) * (x2 - x1) / denom
            if x_int > x - 1e-18:
                inside = not inside
    return inside


def _filter_triangles(
    vertices: Sequence[Tuple[float, float, float]],
    triangles: Sequence[Triangle],
    loops: Sequence[Sequence[Point]],
) -> List[Triangle]:
    if not triangles or not loops:
        return list(triangles)

    loops_native = [list(loop) for loop in loops]
    filtered: List[Triangle] = []
    for tri in triangles:
        a = vertices[tri[0]]
        b = vertices[tri[1]]
        c = vertices[tri[2]]
        cx = (a[0] + b[0] + c[0]) / 3.0
        cy = (a[1] + b[1] + c[1]) / 3.0
        count = 0
        for loop in loops_native:
            if _point_in_loop((cx, cy), loop):
                count += 1
        if count % 2 == 1:
            filtered.append(tri)
    return filtered


def triangulate(
    outer: Sequence[Point],
    inner_loops: Sequence[Sequence[Point]],
    *,
    maxh: float | None,
    quality: str,
    enforce_constraints: bool,
):
    if not CLI_PATH.exists():
        raise RuntimeError(f"cdt_triangulate executable not found at {CLI_PATH}")

    global _QUALITY_WARNED
    if quality != "default" and not _QUALITY_WARNED:
        log_path = Path("results_CDT") / "run_CDT.log"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with log_path.open("a", encoding="utf-8") as log_file:
            log_file.write("NOTE: CDT adapter does not expose explicit quality controls; using defaults.\n")
        _QUALITY_WARNED = True

    loops: List[Sequence[Point]] = [outer]
    loops.extend(inner_loops)
    points, edges = _build_edges(loops)

    max_area = -1.0
    if maxh is not None and maxh > 0:
        max_area = TARGET_AREA_FACTOR * maxh * maxh

    with tempfile.TemporaryDirectory() as tmpdir_str:
        tmpdir = Path(tmpdir_str)
        input_path = tmpdir / "input.txt"
        output_path = tmpdir / "output.txt"
        _write_input(input_path, points, edges)

        cmd = [str(CLI_PATH), "--input", str(input_path), "--output", str(output_path)]
        if max_area > 0:
            cmd.extend(["--max-area", f"{max_area}"])
        if enforce_constraints:
            cmd.append("--enforce")
        cmd.extend(["--quality", quality])

        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(
                "cdt_triangulate failed with code "
                f"{result.returncode}: {result.stderr.strip()}"
            )

        pts, tris, lines = _parse_output(output_path)

    tris = _filter_triangles(pts, tris, loops)

    return pts, tris, lines
