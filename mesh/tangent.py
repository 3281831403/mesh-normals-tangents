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


class TangentError(Exception):
    """Raised when a mesh or tangent basis violates the documented contract."""


def face_normal(p0, p1, p2) -> list[float]:
    """Return the unit face normal for the triangle ``(p0, p1, p2)``.

    The winding is counter-clockwise when viewed from the front, so the normal
    is ``normalize(cross(p1 - p0, p2 - p0))``.  A degenerate triangle (zero
    area, collinear points, or coincident points) returns ``[0.0, 0.0, 0.0]``
    as an explicit degenerate marker; NaN values are never returned.

    Conventions: right-handed TBN, UV in [0, 1], tolerance 1e-6.
    """
    raise NotImplementedError("face_normal is not implemented")


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
    raise NotImplementedError("smooth_normals is not implemented")


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
    raise NotImplementedError("compute_tangents is not implemented")


def orthogonalize(tangent, normal) -> list[float]:
    """Project ``tangent`` onto the plane perpendicular to ``normal``.

    This is Gram-Schmidt: subtract the normal component from the tangent and
    normalize the result.  The returned vector has unit length and its dot
    product with ``normal`` is at most ``1e-6``.  A zero-length projection
    raises :class:`TangentError`.

    Conventions: right-handed TBN, UV in [0, 1], tolerance 1e-6.
    """
    raise NotImplementedError("orthogonalize is not implemented")


def handedness(tangent, bitangent, normal) -> float:
    """Return the sign of the tangent-space determinant, either ``1.0`` or ``-1.0``.

    The sign is that of ``dot(cross(tangent, bitangent), normal)``.  A
    right-handed basis returns ``1.0``; a mirrored basis returns ``-1.0``.
    Degenerate input raises :class:`TangentError`.

    Conventions: right-handed TBN, UV in [0, 1], tolerance 1e-6.
    """
    raise NotImplementedError("handedness is not implemented")


def fix_mirrored_uv(mesh) -> dict:
    """Recompute a tangent basis and repair mirrored UV islands.

    When a triangle has negative UV winding, its bitangent is flipped so that
    ``cross(tangent, bitangent)`` follows the geometric normal again, removing
    the black seam caused by a mirrored basis.  The input mesh is not mutated.
    The return value has the same shape as :func:`compute_tangents`::

        {"tangents": [[x, y, z], ...], "bitangents": [[x, y, z], ...]}

    Conventions: right-handed TBN, UV in [0, 1], tolerance 1e-6.
    """
    raise NotImplementedError("fix_mirrored_uv is not implemented")


def validate_tangents(mesh, tangents) -> list[str]:
    """Return a list of contract violations for ``mesh`` and ``tangents``.

    Every tangent must have unit length (within ``1e-6``), be perpendicular to
    its vertex normal (``abs(dot) <= 1e-6``), and form a right-handed basis
    with the UV-derived bitangent.  The returned strings describe each problem;
    an empty list means the tangents are valid.  An empty mesh returns ``[]``.

    Conventions: right-handed TBN, UV in [0, 1], tolerance 1e-6.
    """
    raise NotImplementedError("validate_tangents is not implemented")


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
    raise NotImplementedError("tangent_batch is not implemented")
