import subprocess
import tempfile
from pathlib import Path
from typing import List, Tuple, Optional

# Paths are resolved relative to this adapter file.
ROOT_DIR = Path(__file__).resolve().parent
JIGSAW_DIR = ROOT_DIR / "jigsaw-0.9.14"
JIGSAW_BIN = JIGSAW_DIR / "build" / "src" / "jigsaw"
RESULTS_DIR = ROOT_DIR / "results_JIGSAW"
RUN_LOG_PATH = RESULTS_DIR / "run_JIGSAW.log"
EDGE_TAG = 20  # JIGSAW_EDGE2_TAG

if not JIGSAW_BIN.exists():
    raise FileNotFoundError(f"JIGSAW executable not found at {JIGSAW_BIN}")

RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def _log(message: str) -> None:
    try:
        with RUN_LOG_PATH.open("a", encoding="ascii") as fh:
            fh.write(message.rstrip() + "\n")
    except OSError:
        pass


def _polygon_area(loop: List[Tuple[float, float]]) -> float:
    area = 0.0
    for (x1, y1), (x2, y2) in zip(loop, loop[1:] + loop[:1]):
        area += x1 * y2 - x2 * y1
    return 0.5 * area


def _ensure_orientation(loop: List[Tuple[float, float]], ccw: bool) -> List[Tuple[float, float]]:
    area = _polygon_area(loop)
    if ccw and area < 0:
        return list(reversed(loop))
    if not ccw and area > 0:
        return list(reversed(loop))
    return loop


def _format_float(value: float) -> str:
    return f"{value:.17g}"


def _write_geom_file(
    path: Path,
    loops: List[List[Tuple[float, float]]],
) -> Tuple[List[Tuple[float, float]], List[Tuple[int, int]]]:
    point_index: dict[Tuple[float, float], int] = {}
    points: List[Tuple[float, float]] = []
    edges: List[Tuple[int, int]] = []
    edge_tags: List[int] = []
    bound_records: List[Tuple[int, int, int]] = []  # (boundary_id, edge_index, kind)

    for boundary_id, loop in enumerate(loops):
        if boundary_id == 0:
            oriented = _ensure_orientation(loop, ccw=True)
        else:
            oriented = _ensure_orientation(loop, ccw=_polygon_area(loop) > 0.0)
        indices: List[int] = []
        for pt in oriented:
            if pt not in point_index:
                point_index[pt] = len(points)
                points.append(pt)
            indices.append(point_index[pt])
        start_edge = len(edges)
        n = len(indices)
        for i in range(n):
            a = indices[i]
            b = indices[(i + 1) % n]
            edges.append((a, b))
            edge_tags.append(boundary_id)
        for local_offset in range(n):
            bound_records.append((boundary_id, start_edge + local_offset, EDGE_TAG))

    with path.open("w", encoding="ascii") as fh:
        fh.write("mshid=1\n")
        fh.write("ndims=2\n")
        fh.write(f"point={len(points)}\n")
        for x, y in points:
            fh.write(f"{_format_float(x)};{_format_float(y)};0\n")
        fh.write(f"edge2={len(edges)}\n")
        for (a, b), tag in zip(edges, edge_tags):
            fh.write(f"{a};{b};{tag}\n")
        fh.write(f"bound={len(bound_records)}\n")
        for tag, edge_idx, kind in bound_records:
            fh.write(f"{tag};{edge_idx};{kind}\n")

    return points, edges


def _write_jig_file(
    path: Path,
    geom_file: Path,
    mesh_file: Path,
    *,
    maxh: Optional[float],
    quality: str,
) -> None:
    lines = [
        f"GEOM_FILE = {geom_file}",
        f"MESH_FILE = {mesh_file}",
        "MESH_DIMS = 2",
    ]
    if maxh is not None:
        hval = max(maxh, 1e-6)
        lines.append("HFUN_SCAL = ABSOLUTE")
        lines.append(f"HFUN_HMAX = {_format_float(hval)}")
    else:
        lines.append("HFUN_SCAL = RELATIVE")
    if quality == "moderate":
        lines.append("OPTM_KERN = CVT+DQDX")
        lines.append("OPTM_ITER = 32")
        lines.append("MESH_RAD2 = 1.5")
    with path.open("w", encoding="ascii") as fh:
        fh.write("\n".join(lines))


def _run_jigsaw(cfg_path: Path) -> None:
    result = subprocess.run(
        [str(JIGSAW_BIN), str(cfg_path)],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
        text=True,
    )
    if result.returncode != 0:
        _log(f"JIGSAW failed with code {result.returncode}: {result.stderr.strip()}")
        raise RuntimeError(f"JIGSAW execution failed (code {result.returncode})")


def _parse_msh(mesh_path: Path) -> Tuple[List[Tuple[float, float, float]], List[Tuple[int, int, int]], List[Tuple[int, int]]]:
    points: List[Tuple[float, float, float]] = []
    triangles: List[Tuple[int, int, int]] = []
    lines: List[Tuple[int, int]] = []

    current_section: Optional[str] = None
    remaining = 0
    ndims = 2

    def _set_section(section: str, count: int) -> None:
        nonlocal current_section, remaining
        current_section = section
        remaining = count

    with mesh_path.open("r", encoding="ascii") as fh:
        for raw_line in fh:
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            if "=" in line:
                key, rest = line.split("=", 1)
                key = key.strip().lower()
                value_part = rest.split(";")[0].strip()
                if key == "ndims":
                    ndims = int(value_part)
                    continue
                if key in {"point", "tria3", "edge2"}:
                    _set_section(key, int(value_part))
                else:
                    current_section = None
                continue
            if not current_section or remaining <= 0:
                continue
            tokens = [tok for tok in line.replace(",", ";").split(";") if tok]
            if current_section == "point":
                if len(tokens) < ndims:
                    raise ValueError("Malformed POINT record")
                coords = tuple(float(tokens[i]) for i in range(ndims))
                points.append((coords[0], coords[1], 0.0))
            elif current_section == "tria3":
                if len(tokens) < 3:
                    raise ValueError("Malformed TRIA3 record")
                triangles.append(tuple(int(tokens[i]) for i in range(3)))
            elif current_section == "edge2":
                if len(tokens) < 2:
                    raise ValueError("Malformed EDGE2 record")
                lines.append((int(tokens[0]), int(tokens[1])))
            remaining -= 1
            if remaining == 0:
                current_section = None

    return points, triangles, lines


def triangulate(
    outer: List[Tuple[float, float]],
    inner_loops: List[List[Tuple[float, float]]],
    *,
    maxh: Optional[float],
    quality: str,
    enforce_constraints: bool,
):
    if quality not in {"default", "moderate"}:
        raise ValueError(f"Unsupported quality setting: {quality}")
    if not enforce_constraints:
        _log("JIGSAW enforces constraint edges; ignoring enforce_constraints=False")

    all_loops = [outer] + inner_loops
    for loop in all_loops:
        if len(loop) < 3:
            raise ValueError("Each polygon loop must contain at least 3 vertices")

    with tempfile.TemporaryDirectory(prefix="jigsaw_adapter_", dir=str(ROOT_DIR)) as tmpdir:
        tmpdir_path = Path(tmpdir)
        geom_path = tmpdir_path / "geom.msh"
        mesh_path = tmpdir_path / "mesh.msh"
        cfg_path = tmpdir_path / "job.jig"

        _write_geom_file(geom_path, all_loops)
        _write_jig_file(cfg_path, geom_path, mesh_path, maxh=maxh, quality=quality)
        _run_jigsaw(cfg_path)

        points, triangles, lines = _parse_msh(mesh_path)

    return points, triangles, lines


_log("adapter_jigsaw initialised; quality='moderate' enables OPTM_KERN CVT+DQDX and relaxed radius-edge targets.")
