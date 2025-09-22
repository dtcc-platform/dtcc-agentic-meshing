import math
import os
import subprocess
import tempfile
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

Point2D = Tuple[float, float]
Point3D = Tuple[float, float, float]
Tri = Tuple[int, int, int]
Seg = Tuple[int, int]

ROOT = Path(__file__).resolve().parent
GEOGRAM_CDT = ROOT / "geogram_cdt"
GEOGRAM_LIBDIR = ROOT / "geogram_1.9.6" / "build" / "lib"


def _dedupe_loop(loop: Sequence[Point2D], tol: float = 1e-12) -> List[Point2D]:
    if not loop:
        return []
    cleaned = list(loop)
    if len(cleaned) > 1:
        x0, y0 = cleaned[0]
        x1, y1 = cleaned[-1]
        if abs(x0 - x1) <= tol and abs(y0 - y1) <= tol:
            cleaned = cleaned[:-1]
    return [(float(x), float(y)) for x, y in cleaned]


def _write_input_file(
    path: Path,
    outer: Sequence[Point2D],
    inner_loops: Sequence[Sequence[Point2D]],
    maxh: Optional[float],
    quality: str,
    enforce_constraints: bool,
) -> None:
    maxh_val = -1.0 if maxh is None or maxh <= 0.0 else float(maxh)
    with path.open("w", encoding="ascii") as f:
        f.write(f"maxh {maxh_val}\n")
        f.write(f"quality {quality}\n")
        f.write(f"enforce_constraints {1 if enforce_constraints else 0}\n")
        outer_pts = _dedupe_loop(outer)
        f.write(f"outer {len(outer_pts)}\n")
        for x, y in outer_pts:
            f.write(f"{x} {y}\n")
        f.write(f"inner_count {len(inner_loops)}\n")
        for loop in inner_loops:
            pts = _dedupe_loop(loop)
            f.write(f"inner {len(pts)}\n")
            for x, y in pts:
                f.write(f"{x} {y}\n")


def _parse_output_file(path: Path) -> Tuple[List[Point3D], List[Tri], List[Seg]]:
    with path.open("r", encoding="ascii") as f:
        lines = [ln.strip() for ln in f if ln.strip()]
    cursor = 0
    def read_section(expected: str) -> Tuple[str, int]:
        nonlocal cursor
        if cursor >= len(lines):
            raise RuntimeError(f"Unexpected end of output while expecting '{expected}'")
        header = lines[cursor].split()
        cursor += 1
        if len(header) != 2 or header[0] != expected:
            raise RuntimeError(f"Expected header '{expected} <count>' but got '{lines[cursor-1]}'")
        return header[0], int(header[1])

    _, num_points = read_section("points")
    points_xy: List[Point2D] = []
    for _ in range(num_points):
        if cursor >= len(lines):
            raise RuntimeError("Unexpected end of output while reading points")
        parts = lines[cursor].split()
        cursor += 1
        if len(parts) != 2:
            raise RuntimeError(f"Point line expected 2 values, got: '{lines[cursor-1]}'")
        points_xy.append((float(parts[0]), float(parts[1])))

    _, num_tris = read_section("triangles")
    tris: List[Tri] = []
    for _ in range(num_tris):
        if cursor >= len(lines):
            raise RuntimeError("Unexpected end of output while reading triangles")
        parts = lines[cursor].split()
        cursor += 1
        if len(parts) != 3:
            raise RuntimeError(f"Triangle line expected 3 values, got: '{lines[cursor-1]}'")
        tris.append((int(parts[0]), int(parts[1]), int(parts[2])))

    _, num_lines = read_section("lines")
    segs: List[Seg] = []
    for _ in range(num_lines):
        if cursor >= len(lines):
            raise RuntimeError("Unexpected end of output while reading lines")
        parts = lines[cursor].split()
        cursor += 1
        if len(parts) != 2:
            raise RuntimeError(f"Line segment expected 2 values, got: '{lines[cursor-1]}'")
        segs.append((int(parts[0]), int(parts[1])))

    points_xyz = [(x, y, 0.0) for x, y in points_xy]
    return points_xyz, tris, segs


def triangulate(
    outer: Sequence[Point2D],
    inner_loops: Sequence[Sequence[Point2D]],
    *,
    maxh: Optional[float],
    quality: str,
    enforce_constraints: bool,
) -> Tuple[List[Point3D], List[Tri], List[Seg]]:
    if not GEOGRAM_CDT.exists():
        raise RuntimeError(f"geogram_cdt binary not found at {GEOGRAM_CDT}")

    if maxh is not None and maxh > 0.0 and maxh < 2.0:
        raise RuntimeError(f"FeatureMissing: requested maxh={maxh} below supported minimum 2.0")

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir)
        input_path = tmpdir_path / "input.txt"
        output_path = tmpdir_path / "output.txt"
        _write_input_file(input_path, outer, inner_loops, maxh, quality, enforce_constraints)

        env = os.environ.copy()
        lib_path = str(GEOGRAM_LIBDIR)
        env_ld = "LD_LIBRARY_PATH"
        env[env_ld] = f"{lib_path}:{env.get(env_ld, '')}" if env.get(env_ld) else lib_path

        cmd = [str(GEOGRAM_CDT), str(input_path), str(output_path)]
        result = subprocess.run(cmd, capture_output=True, text=True, env=env)
        if result.returncode != 0:
            raise RuntimeError(
                "geogram_cdt failed with code {}\nSTDOUT:{}\nSTDERR:{}".format(
                    result.returncode, result.stdout, result.stderr
                )
            )
        if not output_path.exists():
            raise RuntimeError("geogram_cdt did not produce output file")

        return _parse_output_file(output_path)
