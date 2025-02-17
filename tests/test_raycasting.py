"""
Test suite for ray casting operations.
Tests the ray-triangle intersection algorithm for various scenarios.
"""

import numpy as np
from src.kernel.raycasting_mod import check_ray_triangle_intersection, EPSILON


def test_direct_hit():
    """Test ray hitting triangle center."""
    # Triangle in XY plane at z=2
    v0 = np.array([-1.0, -1.0, 2.0])
    v1 = np.array([1.0, -1.0, 2.0])
    v2 = np.array([0.0, 1.0, 2.0])

    # Ray from origin looking down -Z
    ray_origin = np.array([0.0, 0.0, 0.0])
    ray_direction = np.array([0.0, 0.0, -1.0])  # Looking down -Z

    hit, t, u, v = check_ray_triangle_intersection(
        ray_origin, ray_direction, v0, v1, v2
    )
    assert hit
    assert abs(t - 2.0) < EPSILON  # Should hit at z=2
    assert u + v < 1.0  # Inside triangle


def test_parallel_miss():
    """Test ray parallel to triangle plane."""
    v0 = np.array([-1.0, -1.0, 2.0])
    v1 = np.array([1.0, -1.0, 2.0])
    v2 = np.array([0.0, 1.0, 2.0])

    ray_origin = np.array([0.0, 0.0, 0.0])
    ray_direction = np.array([1.0, 0.0, 0.0])  # Parallel to XY plane

    hit, _, _, _ = check_ray_triangle_intersection(
        ray_origin, ray_direction, v0, v1, v2
    )
    assert not hit


def test_backface_culling():
    """Test ray hitting triangle from behind."""
    v0 = np.array([-1.0, -1.0, 2.0])
    v1 = np.array([1.0, -1.0, 2.0])
    v2 = np.array([0.0, 1.0, 2.0])

    # Ray coming from behind triangle looking down -Z
    ray_origin = np.array([0.0, 0.0, 3.0])
    ray_direction = np.array([0.0, 0.0, -1.0])  # Looking down -Z

    hit, _, _, _ = check_ray_triangle_intersection(
        ray_origin, ray_direction, v0, v1, v2
    )
    assert (
        not hit
    )  # Should miss since triangle normal faces +Z and we're looking down -Z


def test_edge_hit():
    """Test ray hitting triangle edge."""
    v0 = np.array([-1.0, -1.0, 2.0])
    v1 = np.array([1.0, -1.0, 2.0])
    v2 = np.array([0.0, 1.0, 2.0])

    # Ray hitting edge between v0 and v1 looking down -Z
    ray_origin = np.array([0.0, -1.0, 0.0])
    ray_direction = np.array([0.0, 0.0, -1.0])  # Looking down -Z

    hit, _, u, v = check_ray_triangle_intersection(
        ray_origin, ray_direction, v0, v1, v2
    )
    assert hit
    assert abs(v) < 0.1  # Should be close to edge (v ≈ 0)


def test_vertex_hit():
    """Test ray hitting triangle vertex."""
    v0 = np.array([-1.0, -1.0, 2.0])
    v1 = np.array([1.0, -1.0, 2.0])
    v2 = np.array([0.0, 1.0, 2.0])

    # Ray hitting vertex v0 looking down -Z
    ray_origin = np.array([-1.0, -1.0, 0.0])
    ray_direction = np.array([0.0, 0.0, -1.0])  # Looking down -Z

    hit, _, u, v = check_ray_triangle_intersection(
        ray_origin, ray_direction, v0, v1, v2
    )
    assert hit
    assert u < EPSILON and v < EPSILON  # Should be at first vertex


def test_miss_outside():
    """Test ray missing triangle entirely."""
    v0 = np.array([-1.0, -1.0, 2.0])
    v1 = np.array([1.0, -1.0, 2.0])
    v2 = np.array([0.0, 1.0, 2.0])

    # Ray passing outside triangle looking down -Z
    ray_origin = np.array([2.0, 2.0, 0.0])
    ray_direction = np.array([0.0, 0.0, -1.0])  # Looking down -Z

    hit, _, _, _ = check_ray_triangle_intersection(
        ray_origin, ray_direction, v0, v1, v2
    )
    assert not hit


def test_behind_ray():
    """Test ray intersection with triangle in front."""
    # Triangle at z=-2 with counter-clockwise winding when viewed from front
    v0 = np.array([-1.0, -1.0, -2.0])
    v1 = np.array([0.0, 1.0, -2.0])  # Top
    v2 = np.array([1.0, -1.0, -2.0])  # Right

    # Ray looking down -Z
    ray_origin = np.array([0.0, 0.0, 0.0])
    ray_direction = np.array([0.0, 0.0, -1.0])  # Looking down -Z

    hit, t, _, _ = check_ray_triangle_intersection(
        ray_origin, ray_direction, v0, v1, v2
    )
    assert not hit  # Should miss - triangle winding is clockwise when looking down -Z


def test_glancing_hit():
    """Test ray hitting triangle at a very shallow angle."""
    v0 = np.array([-1.0, -1.0, 2.0])
    v1 = np.array([1.0, -1.0, 2.0])
    v2 = np.array([0.0, 1.0, 2.0])

    # Ray almost parallel to triangle, looking down -Z
    ray_origin = np.array([-0.1, 0.0, 0.0])
    ray_direction = np.array([0.1, 0.0, -1.0])  # Looking down -Z with slight X
    ray_direction = ray_direction / np.linalg.norm(ray_direction)

    hit, _, _, _ = check_ray_triangle_intersection(
        ray_origin, ray_direction, v0, v1, v2
    )
    assert hit  # Should still detect glancing hit


def test_degenerate_triangle():
    """Test ray against degenerate (zero-area) triangle."""
    v0 = np.array([0.0, 0.0, 2.0])
    v1 = np.array([0.0, 0.0, 2.0])  # Same as v0
    v2 = np.array([1.0, 0.0, 2.0])

    # Ray looking down -Z
    ray_origin = np.array([0.0, 0.0, 0.0])
    ray_direction = np.array([0.0, 0.0, -1.0])  # Looking down -Z

    hit, _, _, _ = check_ray_triangle_intersection(
        ray_origin, ray_direction, v0, v1, v2
    )
    assert not hit  # Should reject degenerate triangle


def test_barycentric_coordinates():
    """Test accuracy of barycentric coordinates at intersection."""
    v0 = np.array([-1.0, -1.0, 2.0])
    v1 = np.array([1.0, -1.0, 2.0])
    v2 = np.array([0.0, 1.0, 2.0])

    # Ray hitting center of triangle looking down -Z
    ray_origin = np.array([0.0, 0.0, 0.0])
    ray_direction = np.array([0.0, 0.0, -1.0])  # Looking down -Z

    hit, _, u, v = check_ray_triangle_intersection(
        ray_origin, ray_direction, v0, v1, v2
    )
    assert hit
    # Allow some tolerance for barycentric coordinates
    assert u + v < 1.0  # Inside triangle
    assert u > 0.0 and v > 0.0  # Not on edges
