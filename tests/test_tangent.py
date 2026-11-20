"""Regression tests for mesh normal and tangent generation.

The implementation in ``mesh/tangent.py`` is intentionally all stubs, so this
suite starts red: every test calls into at least one stubbed function and fails
with ``NotImplementedError`` until the ten defects are fixed.
"""

import math

import pytest

from mesh.tangent import (
    TangentError,
    compute_tangents,
    face_normal,
    fix_mirrored_uv,
    handedness,
    orthogonalize,
    smooth_normals,
    tangent_batch,
    validate_tangents,
)

TOL = 1e-6


def _mesh(positions, normals, uvs, indices):
    return {
        "positions": positions,
        "normals": normals,
        "uvs": uvs,
        "indices": indices,
    }


def _unit_triangle():
    return _mesh(
        positions=[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
        normals=[[0.0, 0.0, 1.0], [0.0, 0.0, 1.0], [0.0, 0.0, 1.0]],
        uvs=[[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]],
        indices=[0, 1, 2],
    )


def _rotated_unit_triangle():
    """A unit triangle whose +U tangent points along +Y instead of +X."""
    return _mesh(
        positions=[[0.0, 0.0, 0.0], [0.0, 1.0, 0.0], [-1.0, 0.0, 0.0]],
        normals=[[0.0, 0.0, 1.0], [0.0, 0.0, 1.0], [0.0, 0.0, 1.0]],
        uvs=[[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]],
        indices=[0, 1, 2],
    )


def _hard_edge_mesh():
    """Two triangles sharing edge 0-1 at a 90 degree dihedral angle."""
    return _mesh(
        positions=[
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ],
        normals=[
            [0.0, 0.0, 1.0],
            [0.0, 0.0, 1.0],
            [0.0, 0.0, 1.0],
            [0.0, 0.0, 1.0],
        ],
        uvs=[[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]],
        indices=[0, 1, 2, 0, 1, 3],
    )


def _close(actual, expected, tol=TOL):
    return len(actual) == len(expected) and all(
        abs(a - b) <= tol for a, b in zip(actual, expected)
    )


def _length(vector):
    return math.sqrt(sum(component * component for component in vector))


def _dot(left, right):
    return sum(a * b for a, b in zip(left, right))


def test_face_normal_is_normalized():
    normal = face_normal([0.0, 0.0, 0.0], [10.0, 0.0, 0.0], [0.0, 10.0, 0.0])

    assert normal == pytest.approx([0.0, 0.0, 1.0], abs=TOL)
    assert abs(_length(normal) - 1.0) <= TOL


def test_degenerate_triangle_is_marked_without_nan():
    normal = face_normal([0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0])

    assert normal == pytest.approx([0.0, 0.0, 0.0], abs=TOL)
    assert all(math.isfinite(component) for component in normal)


def test_smooth_normals_respects_angle_threshold():
    mesh = _hard_edge_mesh()
    blended = [0.0, math.sqrt(0.5), math.sqrt(0.5)]

    hard = smooth_normals(mesh, 30.0)
    smooth = smooth_normals(mesh, 180.0)

    assert len(hard) == 4
    for index in (0, 1):
        assert _close(hard[index], [0.0, 0.0, 1.0]) or _close(
            hard[index], [0.0, 1.0, 0.0]
        )
        assert not _close(hard[index], blended, 1e-3)
    for index in (0, 1):
        assert _close(smooth[index], blended, 1e-3)


def test_tangent_uses_uv_not_position_delta():
    mesh = _mesh(
        positions=[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
        normals=[[0.0, 0.0, 1.0], [0.0, 0.0, 1.0], [0.0, 0.0, 1.0]],
        uvs=[[0.0, 1.0], [0.0, 0.0], [1.0, 1.0]],
        indices=[0, 1, 2],
    )

    result = compute_tangents(mesh)

    assert result["tangents"][0] == pytest.approx([0.0, 1.0, 0.0], abs=TOL)
    assert result["bitangents"][0] == pytest.approx([-1.0, 0.0, 0.0], abs=TOL)


def test_tangents_are_orthogonalized():
    tangent = orthogonalize([1.0, 0.0, 1.0], [0.0, 0.0, 1.0])

    assert tangent == pytest.approx([1.0, 0.0, 0.0], abs=TOL)
    assert abs(_dot(tangent, [0.0, 0.0, 1.0])) <= TOL
    assert abs(_length(tangent) - 1.0) <= TOL


def test_handedness_detects_flip():
    assert handedness([1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]) == pytest.approx(1.0)
    assert handedness([1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, 1.0]) == pytest.approx(-1.0)


def test_mirrored_uv_flips_bitangent():
    mesh = _mesh(
        positions=[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
        normals=[[0.0, 0.0, 1.0], [0.0, 0.0, 1.0], [0.0, 0.0, 1.0]],
        uvs=[[1.0, 1.0], [1.0, 0.0], [0.0, 1.0]],
        indices=[0, 1, 2],
    )

    result = fix_mirrored_uv(mesh)
    tangent = result["tangents"][0]
    bitangent = result["bitangents"][0]

    assert tangent == pytest.approx([0.0, -1.0, 0.0], abs=TOL)
    assert bitangent == pytest.approx([1.0, 0.0, 0.0], abs=TOL)
    assert handedness(tangent, bitangent, [0.0, 0.0, 1.0]) == pytest.approx(1.0)


def test_smooth_normals_are_area_weighted():
    mesh = _mesh(
        positions=[
            [0.0, 0.0, 0.0],
            [10.0, 0.0, 0.0],
            [0.0, 10.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ],
        normals=[[0.0, 0.0, 1.0] for _ in range(5)],
        uvs=[[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [0.5, 0.5]],
        indices=[0, 1, 2, 0, 3, 4],
    )

    normals = smooth_normals(mesh, 180.0)

    assert len(normals) == 5
    assert normals[0][2] > 0.99
    assert abs(_length(normals[0]) - 1.0) <= TOL


def test_batch_isolates_mesh_buffers():
    result = tangent_batch(
        [_unit_triangle(), _rotated_unit_triangle()],
        {"fix_mirrored": True},
    )

    assert len(result["tangents"]) == 2
    assert result["tangents"][0][0] == pytest.approx([1.0, 0.0, 0.0], abs=TOL)
    assert result["tangents"][1][0] == pytest.approx([0.0, 1.0, 0.0], abs=TOL)

    result["tangents"][0][0][0] = 42.0
    assert result["tangents"][1][0][0] != 42.0


def test_batch_rejects_mesh_with_invalid_index():
    bad = _unit_triangle()
    bad["indices"] = [0, 1, 9]
    valid = _unit_triangle()

    with pytest.raises(TangentError):
        tangent_batch([valid, bad], {})


def test_unit_triangle_normal_points_positive_z():
    normal = face_normal([0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0])

    assert normal == pytest.approx([0.0, 0.0, 1.0], abs=TOL)


def test_zero_angle_disables_smoothing():
    mesh = _hard_edge_mesh()

    normals = smooth_normals(mesh, 0.0)

    assert len(normals) == 4
    assert _close(normals[2], [0.0, 0.0, 1.0])
    assert _close(normals[3], [0.0, 1.0, 0.0])
    for index in (0, 1):
        assert _close(normals[index], [0.0, 0.0, 1.0]) or _close(
            normals[index], [0.0, 1.0, 0.0]
        )


def test_empty_mesh_returns_empty_tangents():
    mesh = _mesh(positions=[], normals=[], uvs=[], indices=[])

    result = compute_tangents(mesh)

    assert result == {"tangents": [], "bitangents": []}


def test_compute_tangents_raises_on_out_of_range_index():
    mesh = _unit_triangle()
    mesh["indices"] = [0, 1, 3]

    with pytest.raises(TangentError):
        compute_tangents(mesh)


def test_batch_success_preserves_order():
    first = _unit_triangle()
    second = _rotated_unit_triangle()

    result = tangent_batch([first, second], {})

    assert result["count"] == 2
    assert result["tangents"][0][0] == pytest.approx([1.0, 0.0, 0.0], abs=TOL)
    assert result["tangents"][1][0] == pytest.approx([0.0, 1.0, 0.0], abs=TOL)
    assert validate_tangents(first, result["tangents"][0]) == []
