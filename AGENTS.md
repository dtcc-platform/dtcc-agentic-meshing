# Benchmarking 2D Triangle Mesh Generators — Agent Instructions

# Preamble

In this context software <SOFTWARE_NAME> name is `software_name` with git repo https://github.com/<SOME_AUTHOR>/`software_name`. Replace all incidents of <SOFTWARE_NAME> with `software_name`.

## 0) Goal
Evaluate alternative 2D triangulators (“software X”) for:
- correctness (constraints honored),
- basic quality (no degenerates),
- performance (triangles/second),
- practicality (build friction & API usability).

## 1) Inputs you will receive
- `SOFTWARE_NAME` and official source URL or repo.
- A local file `testcase.txt`:
  - Line 1: exterior polygon as `x1 y1 x2 y2 ...`
  - Lines 2..N: inner polygons (holes / islands) in the same format.
  - Coordinates in a ~500×500 box. **All inner edges must be constraints**.

## 2) What you must produce (artifacts)
Create an output folder (e.g., `results_<SOFTWARE_NAME>/`) containing:

1. **Build log:** `build_<SOFTWARE_NAME>.log`
2. **Meta info:** `meta_<SOFTWARE_NAME>.json` (OS/CPU/versions)
3. **VTU files:**
   - `A_unit_square_default.vtu`
   - `B_unit_square_with_inner_polygon.vtu`
   - `C_city_100.vtu`
   - `D_city_<maxh>.vtu` for each size in the sweep
4. **Benchmarks:**
   - `bench_<SOFTWARE_NAME>.csv`
   - `bench_<SOFTWARE_NAME>.json`
5. **Run log (optional):** `run_<SOFTWARE_NAME>.log` with notable errors/decisions

## 3) Environment & setup
- Use an **isolated environment** (conda env or virtualenv). Containers are OK.
- Prereqs: `git`, C/C++ toolchain, the project’s build system (`cmake`/`meson`/`make`), `python3`, `pip`.
- Install Python libs used by the harness:
  ```bash
  pip install --no-cache-dir meshio numpy
  ```
- The harness will record OS/CPU/RAM/compiler/Python versions into the meta JSON.

## 4) Download, build, install
- Download the **latest stable release** of the tool (or the latest tagged release if no stable tarball).
- Build in **Release** mode using the project’s documented steps.
- Install to a temporary prefix OR run from the build directory.
- Verify with `--version` or an equivalent help/health command.
- If build fails, stop and capture logs in `build_<SOFTWARE_NAME>.log`.

## 5) Use the reference harness (required)
- Save the code in the *Reference Harness* section below as **`mesh_bench_harness.py`**.
- Implement **one adapter** file `adapter_<tool>.py` exposing a single function:

  ```python
  def triangulate(
      outer,                # List[Tuple[float,float]]
      inner_loops,          # List[List[Tuple[float,float]]]
      *,
      maxh,                 # Optional[float]; None = default
      quality,              # "default" | "moderate"
      enforce_constraints,  # bool; honor PSLG edges if supported
  ):
      """
      Returns:
        points_xyz  : List[Tuple[float,float,float]]  # z=0.0
        triangles   : List[Tuple[int,int,int]]        # 0-based CCW
        lines       : List[Tuple[int,int]]            # optional, 0-based
      """
  ```

### Interpreting options inside adapters
- **Constraints:** If the tool supports PSLG/constrained edges, use it.
  If it only accepts “polygon + holes”, that’s acceptable for A/B; for C/D you must honor edges as constraints (split segments if needed). If impossible, note **FeatureMissing** in logs.
- **`maxh` (size):** If the tool uses **area**, set `area ≈ 0.433 * maxh^2`. If it uses a sizing function, provide a uniform field with that target. If unsupported, document and proceed.
- **`quality="moderate"`:** Use a non-extreme minimum angle/aspect setting if available; otherwise defaults.

## 6) Tests the harness will run
- **A:** Unit square, defaults (sanity).
- **B:** Unit square + one irregular inner polygon; edges must appear (constrained).
- **C:** City testcase (outer + many inner loops), `maxh=100`, moderate quality; constrained edges required.
- **D:** Same as C with `maxh` sweep: `100, 50, 20, 10, 5, 2, 1`. Record time, triangle count, triangles/sec.

**Timing** excludes file I/O; the harness times only the meshing call. Each size runs best-of `--repeats` (default 3).

## 7) How to run
```bash
python3 mesh_bench_harness.py \
  --software "<SOFTWARE_NAME>" \
  --adapter adapter_<tool>.py \
  --testcase testcase.txt \
  --outdir results_<SOFTWARE_NAME> \
  --sizes 100 50 20 10 5 2 1 \
  --repeats 3
```

## 8) Quality/validity checks (lightweight)
- Adapters should avoid zero-area elements; triangles must be valid.
- For constraint verification (B/C/D), each input segment should appear as an edge or chain of edges in the output. If the tool flags constrained edges, prefer that; otherwise nearest-node matching is acceptable for inspection.

## 9) Failure policy
- **BuildFailed:** stop; write logs and artifacts produced so far.
- **FeatureMissing:** if constraints or size control aren’t supported, mark in the notes and continue with what’s possible.
- **MeshFailed:** record error for that case/size and continue remaining sizes.
- **TimeOut:** If running the bench takes more than 1 hour kill the process and record error.

## 10) Metrics
- **MeshQuality:** Implement triangle minimum angle distribution, triangle aspect ratio distribution, triangle surface, triangle count and save them into `metrics_<SOFTWARE_NAME>.log`


---

## Reference Harness (`mesh_bench_harness.py`)

```python
#!/usr/bin/env python3
# mesh_bench_harness.py
from __future__ import annotations
import argparse, importlib.util, json, os, platform, time, re
from dataclasses import dataclass, asdict
from typing import List, Tuple, Optional, Dict
import numpy as np
import meshio

XY  = Tuple[float, float]
XYZ = Tuple[float, float, float]
Tri = Tuple[int, int, int]
Seg = Tuple[int, int]
Loop = List[XY]

# ---------- IO & geometry ----------

def tokens_to_pairs(tokens: List[str]) -> Loop:
    if len(tokens) % 2 != 0:
        raise ValueError(f"Odd number of coordinates ({len(tokens)}); expected x y x y ...")
    it = iter(tokens)
    return [(float(x), float(y)) for x, y in zip(it, it)]

def dedupe_closing_vertex(coords: Loop, tol: float = 1e-12) -> Loop:
    if len(coords) >= 2:
        x0, y0 = coords[0]; x1, y1 = coords[-1]
        if abs(x0 - x1) <= tol and abs(y0 - y1) <= tol:
            return coords[:-1]
    return coords

def read_testcase(path: str) -> Tuple[Loop, List[Loop]]:
    lines: List[str] = []
    with open(path, "r") as f:
        for raw in f:
            line = raw.strip()
            if line:
                lines.append(line)
    if not lines:
        raise ValueError("Empty testcase file")
    def parse_line(line: str) -> Loop:
        tokens = re.split(r"\s+", line.strip())
        pts = dedupe_closing_vertex(tokens_to_pairs(tokens))
        if len(pts) < 3:
            raise ValueError("Polygon has < 3 vertices")
        return pts
    outer = parse_line(lines[0])
    inners = [parse_line(ln) for ln in lines[1:]]
    return outer, inners

def unit_square() -> Loop:
    return [(0.0,0.0),(1.0,0.0),(1.0,1.0),(0.0,1.0)]

def irregular_inner_poly() -> Loop:
    return [(0.2,0.2),(0.8,0.25),(0.75,0.7),(0.5,0.85),(0.25,0.75)]

def write_vtu(points: List[XYZ], tris: List[Tri], lines: List[Seg], path: str,
              cell_data: Optional[Dict[str, List[np.ndarray]]] = None) -> None:
    pts = np.asarray(points, dtype=np.float64)
    tri_cells = np.asarray(tris, dtype=np.int32) if tris else np.zeros((0,3), dtype=np.int32)
    cells = [("triangle", tri_cells)]
    if lines:
        line_cells = np.asarray(lines, dtype=np.int32)
        cells.append(("line", line_cells))
        if cell_data and "region" in cell_data:
            r = cell_data["region"]
            if len(r) == 1:
                r = [r[0], np.zeros(line_cells.shape[0], dtype=np.int32)]
                cell_data = {"region": r}
    meshio.Mesh(points=pts, cells=cells, cell_data=cell_data or {}).write(path)

def count_tris(tris: List[Tri]) -> int:
    return int(len(tris))

# ---------- adapter loader ----------

def load_adapter(path: str):
    spec = importlib.util.spec_from_file_location("adapter_module", path)
    mod = importlib.util.module_from_spec(spec)  # type: ignore
    assert spec and spec.loader
    spec.loader.exec_module(mod)                 # type: ignore
    if not hasattr(mod, "triangulate"):
        raise RuntimeError("Adapter missing triangulate(...)")
    return mod

# ---------- timing ----------

def time_call(fn, *args, repeats: int = 1, warmup: bool = True, **kwargs):
    if warmup:
        fn(*args, **kwargs)
    best = None
    last_result = None
    for _ in range(max(1, repeats)):
        t0 = time.perf_counter()
        result = fn(*args, **kwargs)
        dt = time.perf_counter() - t0
        if best is None or dt < best:
            best = dt
            last_result = result
    return last_result, float(best)

# ---------- bench protocol ----------

@dataclass
class BenchRow:
    software: str
    case: str
    maxh: Optional[float]
    time_s: float
    num_triangles: int
    triangles_per_s: float
    notes: str = ""

def run_case_A(software: str, adapter, outdir: str) -> BenchRow:
    outer = unit_square()
    pts, tris, lines = adapter.triangulate(outer, [], maxh=None, quality="default", enforce_constraints=False)
    write_vtu(pts, tris, lines, os.path.join(outdir, "A_unit_square_default.vtu"))
    n = count_tris(tris)
    return BenchRow(software, "A", None, 0.0, n, 0.0, "sanity")

def run_case_B(software: str, adapter, outdir: str) -> BenchRow:
    outer = unit_square()
    inner = [irregular_inner_poly()]
    pts, tris, lines = adapter.triangulate(outer, inner, maxh=None, quality="default", enforce_constraints=True)
    write_vtu(pts, tris, lines, os.path.join(outdir, "B_unit_square_with_inner_polygon.vtu"))
    n = count_tris(tris)
    return BenchRow(software, "B", None, 0.0, n, 0.0, "sanity")

def run_case_C(software: str, adapter, outdir: str, testcase: str) -> BenchRow:
    outer, inners = read_testcase(testcase)
    (pts, tris, lines), dt = time_call(
        adapter.triangulate, outer, inners, maxh=100.0, quality="moderate", enforce_constraints=True,
        repeats=3, warmup=True
    )
    write_vtu(pts, tris, lines, os.path.join(outdir, "C_city_100.vtu"))
    n = count_tris(tris)
    return BenchRow(software, "C", 100.0, dt, n, (n/dt if dt > 0 else 0.0))

def run_case_D(software: str, adapter, outdir: str, testcase: str, sizes: List[float], repeats: int) -> List[BenchRow]:
    outer, inners = read_testcase(testcase)
    rows: List[BenchRow] = []
    for h in sizes:
        try:
            (pts, tris, lines), dt = time_call(
                adapter.triangulate, outer, inners,
                maxh=float(h), quality="moderate", enforce_constraints=True,
                repeats=repeats, warmup=True
            )
            tag = int(h) if float(h).is_integer() else h
            write_vtu(pts, tris, lines, os.path.join(outdir, f"D_city_{tag}.vtu"))
            n = count_tris(tris)
            rows.append(BenchRow(software, "D", float(h), dt, n, (n/dt if dt > 0 else 0.0)))
        except Exception as e:
            rows.append(BenchRow(software, "D", float(h), -1.0, -1, 0.0, notes=f"FAILED: {e}"))
    return rows

# ---------- metadata & reporting ----------

def system_meta(software: str, adapter_path: str) -> Dict:
    return {
        "software": software,
        "adapter_path": os.path.abspath(adapter_path),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "cpu_count": os.cpu_count(),
    }

def write_reports(software: str, outdir: str, rows: List[BenchRow]) -> None:
    os.makedirs(outdir, exist_ok=True)
    csv_path = os.path.join(outdir, f"bench_{software}.csv")
    with open(csv_path, "w") as f:
        f.write("software,case,maxh,time_s,num_triangles,triangles_per_s,notes\n")
        for r in rows:
            f.write(f"{r.software},{r.case},{'' if r.maxh is None else r.maxh},{r.time_s},{r.num_triangles},{r.triangles_per_s},{r.notes}\n")
    json_path = os.path.join(outdir, f"bench_{software}.json")
    with open(json_path, "w") as f:
        json.dump([asdict(r) for r in rows], f, indent=2)

# ---------- main ----------

def main():
    ap = argparse.ArgumentParser(description="Benchmark 2D triangle mesh generators")
    ap.add_argument("--software", required=True, help="Tool name for reporting")
    ap.add_argument("--adapter",  required=True, help="Path to adapter_<tool>.py implementing triangulate(...)")
    ap.add_argument("--testcase", required=True, help="Path to testcase.txt (outer + inner loops)")
    ap.add_argument("--outdir",   required=True, help="Output directory for VTU and reports")
    ap.add_argument("--sizes", nargs="+", type=float, default=[100,50,20,10,5,2,1], help="Size sweep for case D")
    ap.add_argument("--repeats", type=int, default=3, help="Timing repeats; best-of is reported")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    adapter = load_adapter(args.adapter)

    meta = system_meta(args.software, args.adapter)
    with open(os.path.join(args.outdir, f"meta_{args.software}.json"), "w") as f:
        json.dump(meta, f, indent=2)

    rows: List[BenchRow] = []
    rows.append(run_case_A(args.software, adapter, args.outdir))
    rows.append(run_case_B(args.software, adapter, args.outdir))
    rows.append(run_case_C(args.software, adapter, args.outdir, args.testcase))
    rows.extend(run_case_D(args.software, adapter, args.outdir, args.testcase, args.sizes, args.repeats))

    write_reports(args.software, args.outdir, rows)
    print(f"Done. Reports in {args.outdir}")

if __name__ == "__main__":
    main()
```
