"""Normal and tangent helpers for triangle meshes.

Common contract for every public function below:

* Normals and tangent bases use a right-handed coordinate system, so a valid
  basis satisfies ``cross(tangent, bitangent) == normal``.
* UV coordinates are normalized to the inclusive range ``[0, 1]``.
* Floating-point comparisons use an absolute tolerance of ``1e-6``.

A mesh is a mapping with exactly these keys::

    {
        "positions": [[x, y, z], ...],
        "normals":   [[x, y, z], ...],
        "uvs":       [[u, v], ...],
        "indices":   [i0, i1, i2, ...],
    }

``indices`` is a flat list of triangle corner indices, three per triangle.
"""

import math
from collections import deque

TOL = 1e-6


class TangentError(Exception):
    """Raised when a mesh or tangent basis violates the documented contract."""


# ---------------------------------------------------------------------------
# Small vector helpers (all vectors are plain [x, y, z] lists).
# ---------------------------------------------------------------------------

def _sub(a, b):
    return [a[0] - b[0], a[1] - b[1], a[2] - b[2]]


def _cross(a, b):
    return [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]


def _dot(a, b):
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def _length(v):
    return math.sqrt(_dot(v, v))


def _normalize(v):
    """Return the unit vector for ``v``, or ``None`` if it is degenerate."""
    length = _length(v)
    if length <= TOL:
        return None
    return [component / length for component in v]


def _validate_mesh(mesh):
    """Raise :class:`TangentError` if the mesh violates the structural contract."""
    positions = mesh["positions"]
    normals = mesh["normals"]
    uvs = mesh["uvs"]
    indices = mesh["indices"]
    if not (len(positions) == len(normals) == len(uvs)):
        raise TangentError(
            "attribute length mismatch: positions=%d, normals=%d, uvs=%d"
            % (len(positions), len(normals), len(uvs))
        )
    if len(indices) % 3 != 0:
        raise TangentError(
            "index count %d is not a multiple of three" % len(indices)
        )
    count = len(positions)
    for index in indices:
        if not isinstance(index, int) or isinstance(index, bool):
            raise TangentError("index %r is not an integer" % (index,))
        if index < 0 or index >= count:
            raise TangentError(
                "index %d out of range for %d vertices" % (index, count)
            )


def face_normal(p0, p1, p2) -> list[float]:
    """Return the unit face normal for the triangle ``(p0, p1, p2)``.

    The winding is counter-clockwise when viewed from the front, so the normal
    is ``normalize(cross(p1 - p0, p2 - p0))``.  A degenerate triangle (zero
    area, collinear points, or coincident points) returns ``[0.0, 0.0, 0.0]``
    as an explicit degenerate marker; NaN values are never returned.

    Conventions: right-handed TBN, UV in [0, 1], tolerance 1e-6.
    """
    edge1 = _sub(p1, p0)
    edge2 = _sub(p2, p0)
    normal = _cross(edge1, edge2)
    length = _length(normal)
    if length <= TOL:
        # Degenerate triangle: explicit zero marker instead of NaN.
        return [0.0, 0.0, 0.0]
    return [component / length for component in normal]


def _corner_angle(pa, pb, pc):
    """Return the angle in radians at corner ``pa`` of triangle (pa, pb, pc)."""
    u = _sub(pb, pa)
    v = _sub(pc, pa)
    lu = _length(u)
    lv = _length(v)
    if lu <= TOL or lv <= TOL:
        return 0.0
    cosine = max(-1.0, min(1.0, _dot(u, v) / (lu * lv)))
    return math.acos(cosine)


def _orient_consistently(triangles):
    """Flip triangles in place so shared edges are traversed in opposite
    directions, giving the connected mesh a consistent winding."""
    edge_map = {}
    for ti, tri in enumerate(triangles):
        for k in range(3):
            a, b = tri[k], tri[(k + 1) % 3]
            edge_map.setdefault((min(a, b), max(a, b)), []).append((ti, a, b))

    visited = [False] * len(triangles)
    for start in range(len(triangles)):
        if visited[start]:
            continue
        visited[start] = True
        queue = deque([start])
        while queue:
            current = queue.popleft()
            tri = triangles[current]
            for k in range(3):
                a, b = tri[k], tri[(k + 1) % 3]
                for ti, x, y in edge_map[(min(a, b), max(a, b))]:
                    if ti == current or visited[ti]:
                        continue
                    if (x, y) == (a, b):
                        # Same direction along a shared edge: flip the neighbor.
                        triangles[ti][1], triangles[ti][2] = (
                            triangles[ti][2],
                            triangles[ti][1],
                        )
                    visited[ti] = True
                    queue.append(ti)


def smooth_normals(mesh, angle_deg) -> list[list[float]]:
    """Return one smoothed unit normal per mesh position.

    Each incident face contributes a weight of ``face_area * corner_angle``.
    The first incident face seeds the vertex normal and defines its smoothing
    cluster: a face whose normal deviates from that seed by more than
    ``angle_deg`` is excluded, so hard edges are preserved.  An
    ``angle_deg`` of zero or less disables smoothing entirely.

    The returned list has the same length and order as ``mesh["positions"]``.

    Conventions: right-handed TBN, UV in [0, 1], tolerance 1e-6.
    """
    _validate_mesh(mesh)
    positions = mesh["positions"]
    indices = mesh["indices"]

    triangles = [
        [indices[t], indices[t + 1], indices[t + 2]]
        for t in range(0, len(indices), 3)
    ]
    _orient_consistently(triangles)

    # Per-face data: (unit normal, area, [corner angles]) or None if degenerate.
    faces = []
    for tri in triangles:
        p0, p1, p2 = (positions[i] for i in tri)
        edge1 = _sub(p1, p0)
        edge2 = _sub(p2, p0)
        cross = _cross(edge1, edge2)
        cross_len = _length(cross)
        if cross_len <= TOL:
            faces.append(None)
            continue
        normal = [component / cross_len for component in cross]
        area = 0.5 * cross_len
        angles = [
            _corner_angle(p0, p1, p2),
            _corner_angle(p1, p2, p0),
            _corner_angle(p2, p0, p1),
        ]
        faces.append((normal, area, angles))

    incident = [[] for _ in positions]
    for fi, tri in enumerate(triangles):
        if faces[fi] is None:
            continue
        for corner, vertex in enumerate(tri):
            incident[vertex].append((fi, corner))

    result = []
    for vertex in range(len(positions)):
        faces_here = incident[vertex]
        if not faces_here:
            result.append([0.0, 0.0, 0.0])
            continue
        seed_fi = faces_here[0][0]
        seed_normal = faces[seed_fi][0]
        accum = [0.0, 0.0, 0.0]
        for fi, corner in faces_here:
            normal, area, angles = faces[fi]
            if angle_deg > 0.0:
                cosine = max(-1.0, min(1.0, _dot(normal, seed_normal)))
                deviation = math.degrees(math.acos(cosine))
                if deviation > angle_deg:
                    continue  # beyond the smoothing threshold: hard edge
            elif fi != seed_fi:
                continue  # smoothing disabled: only the seed face contributes
            weight = area * angles[corner]
            for k in range(3):
                accum[k] += normal[k] * weight
        smoothed = _normalize(accum)
        result.append(smoothed if smoothed is not None else list(seed_normal))
    return result


def compute_tangents(mesh) -> dict:
    """Derive per-vertex tangents and bitangents from the mesh UVs.

    Tangents follow the +U direction and bitangents follow the +V direction,
    with UVs in ``[0, 1]``.  Tangents are Gram-Schmidt orthogonalized against
    the vertex normals.  The return value is::

        {"tangents": [[x, y, z], ...], "bitangents": [[x, y, z], ...]}

    with one entry per vertex, in vertex order.  Out-of-range indices, a
    non-multiple-of-three index count, or mismatched attribute lengths raise
    :class:`TangentError`.

    Conventions: right-handed TBN, UV in [0, 1], tolerance 1e-6.
    """
    _validate_mesh(mesh)
    positions = mesh["positions"]
    normals = mesh["normals"]
    uvs = mesh["uvs"]
    indices = mesh["indices"]

    count = len(positions)
    tangent_acc = [[0.0, 0.0, 0.0] for _ in range(count)]
    bitangent_acc = [[0.0, 0.0, 0.0] for _ in range(count)]

    for t in range(0, len(indices), 3):
        i0, i1, i2 = indices[t], indices[t + 1], indices[t + 2]
        p0, p1, p2 = positions[i0], positions[i1], positions[i2]
        uv0, uv1, uv2 = uvs[i0], uvs[i1], uvs[i2]

        edge1 = _sub(p1, p0)
        edge2 = _sub(p2, p0)
        du1, dv1 = uv1[0] - uv0[0], uv1[1] - uv0[1]
        du2, dv2 = uv2[0] - uv0[0], uv2[1] - uv0[1]
        det = du1 * dv2 - du2 * dv1
        if abs(det) <= TOL:
            continue  # degenerate UV triangle contributes no tangent
        inv = 1.0 / det
        tangent = [(edge1[k] * dv2 - edge2[k] * dv1) * inv for k in range(3)]
        bitangent = [(edge2[k] * du1 - edge1[k] * du2) * inv for k in range(3)]
        for i in (i0, i1, i2):
            for k in range(3):
                tangent_acc[i][k] += tangent[k]
                bitangent_acc[i][k] += bitangent[k]

    tangents = []
    bitangents = []
    for i in range(count):
        normal = _normalize(normals[i])
        tangent = _orthogonalize_safe(tangent_acc[i], normal)
        bitangent = _orthogonalize_safe(bitangent_acc[i], normal, tangent)
        if bitangent is None and tangent is not None and normal is not None:
            # Right-handed fallback derived from the valid tangent/normal.
            bitangent = _cross(normal, tangent)
        tangents.append(tangent if tangent is not None else [0.0, 0.0, 0.0])
        bitangents.append(bitangent if bitangent is not None else [0.0, 0.0, 0.0])
    return {"tangents": tangents, "bitangents": bitangents}


def _orthogonalize_safe(vector, normal, tangent=None):
    """Gram-Schmidt project ``vector`` against ``normal`` (and ``tangent``).

    Returns a unit vector, or ``None`` when the projection is degenerate.
    """
    projected = list(vector)
    if normal is not None:
        d = _dot(projected, normal)
        for k in range(3):
            projected[k] -= d * normal[k]
    if tangent is not None:
        d = _dot(projected, tangent)
        for k in range(3):
            projected[k] -= d * tangent[k]
    result = _normalize(projected)
    if result is None and normal is not None:
        # Fall back to any unit vector perpendicular to the normal.
        axis = min(range(3), key=lambda k: abs(normal[k]))
        seed = [0.0, 0.0, 0.0]
        seed[axis] = 1.0
        d = _dot(seed, normal)
        fallback = [seed[k] - d * normal[k] for k in range(3)]
        result = _normalize(fallback)
    return result


def orthogonalize(tangent, normal) -> list[float]:
    """Project ``tangent`` onto the plane perpendicular to ``normal``.

    This is Gram-Schmidt: subtract the normal component from the tangent and
    normalize the result.  The returned vector has unit length and its dot
    product with ``normal`` is at most ``1e-6``.  A zero-length projection
    raises :class:`TangentError`.

    Conventions: right-handed TBN, UV in [0, 1], tolerance 1e-6.
    """
    unit_normal = _normalize(normal)
    if unit_normal is None:
        raise TangentError("normal has zero length")
    d = _dot(tangent, unit_normal)
    projected = [tangent[k] - d * unit_normal[k] for k in range(3)]
    result = _normalize(projected)
    if result is None:
        raise TangentError("tangent is parallel to the normal")
    return result


def handedness(tangent, bitangent, normal) -> float:
    """Return the sign of the tangent-space determinant, either ``1.0`` or ``-1.0``.

    The sign is that of ``dot(cross(tangent, bitangent), normal)``.  A
    right-handed basis returns ``1.0``; a mirrored basis returns ``-1.0``.
    Degenerate input raises :class:`TangentError`.

    Conventions: right-handed TBN, UV in [0, 1], tolerance 1e-6.
    """
    det = _dot(_cross(tangent, bitangent), normal)
    if abs(det) <= TOL:
        raise TangentError("degenerate tangent basis has no handedness")
    return 1.0 if det > 0.0 else -1.0


def fix_mirrored_uv(mesh) -> dict:
    """Recompute a tangent basis and repair mirrored UV islands.

    When a triangle has negative UV winding, its bitangent is flipped so that
    ``cross(tangent, bitangent)`` follows the geometric normal again, removing
    the black seam caused by a mirrored basis.  The input mesh is not mutated.
    The return value has the same shape as :func:`compute_tangents`::

        {"tangents": [[x, y, z], ...], "bitangents": [[x, y, z], ...]}

    Conventions: right-handed TBN, UV in [0, 1], tolerance 1e-6.
    """
    _validate_mesh(mesh)
    uvs = mesh["uvs"]
    indices = mesh["indices"]

    result = compute_tangents(mesh)

    # Accumulate the UV winding sign per vertex.
    winding = [0.0] * len(mesh["positions"])
    for t in range(0, len(indices), 3):
        i0, i1, i2 = indices[t], indices[t + 1], indices[t + 2]
        uv0, uv1, uv2 = uvs[i0], uvs[i1], uvs[i2]
        du1, dv1 = uv1[0] - uv0[0], uv1[1] - uv0[1]
        du2, dv2 = uv2[0] - uv0[0], uv2[1] - uv0[1]
        det = du1 * dv2 - du2 * dv1
        for i in (i0, i1, i2):
            winding[i] += det

    for i in range(len(winding)):
        if winding[i] < 0.0:
            bitangent = result["bitangents"][i]
            result["bitangents"][i] = [-component for component in bitangent]
    return result


def validate_tangents(mesh, tangents) -> list[str]:
    """Return a list of contract violations for ``mesh`` and ``tangents``.

    Every tangent must have unit length (within ``1e-6``), be perpendicular to
    its vertex normal (``abs(dot) <= 1e-6``), and form a right-handed basis
    with the UV-derived bitangent.  The returned strings describe each problem;
    an empty list means the tangents are valid.  An empty mesh returns ``[]``.

    Conventions: right-handed TBN, UV in [0, 1], tolerance 1e-6.
    """
    positions = mesh["positions"]
    normals = mesh["normals"]
    if len(positions) == 0:
        return []

    problems = []
    if len(tangents) != len(positions):
        problems.append(
            "tangent count %d does not match vertex count %d"
            % (len(tangents), len(positions))
        )
        return problems

    try:
        derived = compute_tangents(mesh)
    except TangentError as exc:
        return ["invalid mesh: %s" % exc]
    bitangents = derived["bitangents"]

    for i in range(len(positions)):
        tangent = tangents[i]
        length = _length(tangent)
        if abs(length - 1.0) > TOL:
            problems.append(
                "tangent %d is not unit length (|t| = %r)" % (i, length)
            )
        normal = _normalize(normals[i])
        if normal is None:
            problems.append("vertex %d has a zero-length normal" % i)
            continue
        d = _dot(tangent, normal)
        if abs(d) > TOL:
            problems.append(
                "tangent %d is not perpendicular to its normal (dot = %r)"
                % (i, d)
            )
        try:
            if handedness(tangent, bitangents[i], normal) < 0.0:
                problems.append(
                    "tangent %d forms a left-handed (mirrored) basis" % i
                )
        except TangentError:
            problems.append("tangent %d forms a degenerate basis" % i)
    return problems


def tangent_batch(meshes, config) -> dict:
    """Compute tangents for several meshes in input order.

    ``config`` is an optional settings mapping.  Every mesh is validated before
    any result is returned; if one mesh is invalid, the whole batch raises
    :class:`TangentError` and no partial result is exposed.  Each mesh gets an
    independent tangent/bitangent buffer.  The return value is::

        {
            "tangents":  [mesh0_tangents, mesh1_tangents, ...],
            "bitangents": [mesh0_bitangents, mesh1_bitangents, ...],
            "count": len(meshes),
        }

    The order of the outer lists is exactly the input order.

    Conventions: right-handed TBN, UV in [0, 1], tolerance 1e-6.
    """
    config = config or {}
    # Validate every mesh up front: one bad mesh fails the whole batch.
    for mesh in meshes:
        _validate_mesh(mesh)

    tangents = []
    bitangents = []
    for mesh in meshes:
        if config.get("fix_mirrored"):
            result = fix_mirrored_uv(mesh)
        else:
            result = compute_tangents(mesh)
        # Each mesh gets its own freshly allocated buffers.
        tangents.append(result["tangents"])
        bitangents.append(result["bitangents"])
    return {
        "tangents": tangents,
        "bitangents": bitangents,
        "count": len(meshes),
    }
