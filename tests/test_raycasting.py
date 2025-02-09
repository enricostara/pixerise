"""
Test suite for ray casting operations.
Tests the ray-triangle intersection algorithm for various scenarios.
"""

import numpy as np
import pytest
from src.kernel.raycasting_mod import rayIntersectsTriangle, EPSILON

def test_direct_hit():
    """Test ray hitting triangle center."""
    # Triangle in XY plane at z=2
    v0 = np.array([-1.0, -1.0, 2.0])
    v1 = np.array([1.0, -1.0, 2.0])
    v2 = np.array([0.0, 1.0, 2.0])
    
    # Ray from origin pointing straight at triangle
    ray_origin = np.array([0.0, 0.0, 0.0])
    ray_direction = np.array([0.0, 0.0, 1.0])
    
    hit, t, u, v = rayIntersectsTriangle(ray_origin, ray_direction, v0, v1, v2)
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
    
    hit, _, _, _ = rayIntersectsTriangle(ray_origin, ray_direction, v0, v1, v2)
    assert not hit

def test_backface_culling():
    """Test ray hitting triangle from behind."""
    v0 = np.array([-1.0, -1.0, 2.0])
    v1 = np.array([1.0, -1.0, 2.0])
    v2 = np.array([0.0, 1.0, 2.0])
    
    # Ray coming from behind triangle
    ray_origin = np.array([0.0, 0.0, 3.0])
    ray_direction = np.array([0.0, 0.0, -1.0])
    
    hit, _, _, _ = rayIntersectsTriangle(ray_origin, ray_direction, v0, v1, v2)
    assert not hit  # Should be culled

def test_edge_hit():
    """Test ray hitting triangle edge."""
    v0 = np.array([-1.0, -1.0, 2.0])
    v1 = np.array([1.0, -1.0, 2.0])
    v2 = np.array([0.0, 1.0, 2.0])
    
    # Ray hitting edge between v0 and v1
    ray_origin = np.array([0.0, -1.0, 0.0])
    ray_direction = np.array([0.0, 0.0, 1.0])
    
    hit, _, u, v = rayIntersectsTriangle(ray_origin, ray_direction, v0, v1, v2)
    assert hit
    # Allow some tolerance for floating point on edge cases
    assert abs(v) < 0.1  # Should be close to edge (v ≈ 0)

def test_vertex_hit():
    """Test ray hitting triangle vertex."""
    v0 = np.array([-1.0, -1.0, 2.0])
    v1 = np.array([1.0, -1.0, 2.0])
    v2 = np.array([0.0, 1.0, 2.0])
    
    # Ray hitting vertex v0
    ray_origin = np.array([-1.0, -1.0, 0.0])
    ray_direction = np.array([0.0, 0.0, 1.0])
    
    hit, _, u, v = rayIntersectsTriangle(ray_origin, ray_direction, v0, v1, v2)
    assert hit
    assert u < EPSILON and v < EPSILON  # Should be at first vertex

def test_miss_outside():
    """Test ray missing triangle entirely."""
    v0 = np.array([-1.0, -1.0, 2.0])
    v1 = np.array([1.0, -1.0, 2.0])
    v2 = np.array([0.0, 1.0, 2.0])
    
    # Ray passing outside triangle
    ray_origin = np.array([2.0, 2.0, 0.0])
    ray_direction = np.array([0.0, 0.0, 1.0])
    
    hit, _, _, _ = rayIntersectsTriangle(ray_origin, ray_direction, v0, v1, v2)
    assert not hit

def test_behind_ray():
    """Test triangle behind ray origin."""
    v0 = np.array([-1.0, -1.0, -2.0])
    v1 = np.array([1.0, -1.0, -2.0])
    v2 = np.array([0.0, 1.0, -2.0])
    
    ray_origin = np.array([0.0, 0.0, 0.0])
    ray_direction = np.array([0.0, 0.0, 1.0])
    
    hit, _, _, _ = rayIntersectsTriangle(ray_origin, ray_direction, v0, v1, v2)
    assert not hit  # Triangle behind ray

def test_glancing_hit():
    """Test ray hitting triangle at a very shallow angle."""
    v0 = np.array([-1.0, -1.0, 2.0])
    v1 = np.array([1.0, -1.0, 2.0])
    v2 = np.array([0.0, 1.0, 2.0])
    
    # Ray almost parallel to triangle
    ray_origin = np.array([-0.1, 0.0, 0.0])
    ray_direction = np.array([0.1, 0.0, 1.0])
    ray_direction = ray_direction / np.linalg.norm(ray_direction)
    
    hit, _, _, _ = rayIntersectsTriangle(ray_origin, ray_direction, v0, v1, v2)
    assert hit  # Should still detect glancing hit

def test_degenerate_triangle():
    """Test ray against degenerate (zero-area) triangle."""
    v0 = np.array([0.0, 0.0, 2.0])
    v1 = np.array([0.0, 0.0, 2.0])  # Same as v0
    v2 = np.array([1.0, 0.0, 2.0])
    
    ray_origin = np.array([0.0, 0.0, 0.0])
    ray_direction = np.array([0.0, 0.0, 1.0])
    
    hit, _, _, _ = rayIntersectsTriangle(ray_origin, ray_direction, v0, v1, v2)
    assert not hit  # Should reject degenerate triangle

def test_barycentric_coordinates():
    """Test accuracy of barycentric coordinates at intersection."""
    v0 = np.array([-1.0, -1.0, 2.0])
    v1 = np.array([1.0, -1.0, 2.0])
    v2 = np.array([0.0, 1.0, 2.0])
    
    # Ray hitting center of triangle
    ray_origin = np.array([0.0, 0.0, 0.0])
    ray_direction = np.array([0.0, 0.0, 1.0])
    
    hit, _, u, v = rayIntersectsTriangle(ray_origin, ray_direction, v0, v1, v2)
    assert hit
    # Allow some tolerance for barycentric coordinates
    assert u + v < 1.0  # Inside triangle
    assert u > 0.0 and v > 0.0  # Not on edges
