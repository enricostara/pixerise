import pytest
import numpy as np
from src.pixerise import Canvas, ViewPort, Renderer, ShadingMode
from src.scene import Scene, Model, Instance, Camera, ModelInnerGroup

class TestRayCasting:
    @pytest.fixture
    def setup_renderer(self):
        """Setup a basic renderer with canvas and viewport"""
        canvas = Canvas((800, 600))
        viewport = ViewPort((1.6, 1.2), 1.0, canvas)
        return Renderer(canvas, viewport)

    @pytest.fixture
    def setup_basic_scene(self):
        """Setup a scene with a single triangle for basic intersection tests"""
        scene = Scene()
        
        # Create a simple triangle model
        model = Model()
        vertices = np.array([
            [0.0, 1.0, 0.0],   # Top
            [-1.0, -1.0, 0.0], # Bottom left
            [1.0, -1.0, 0.0]   # Bottom right
        ], dtype=np.float32)
        triangles = np.array([[0, 1, 2]], dtype=np.int32)
        model.add_group("default", vertices, triangles)
        scene.add_model("triangle", model)
        
        # Add instance at z=-5 (in front of camera)
        instance = Instance(model="triangle")
        instance.set_translation(0.0, 0.0, -5.0)
        scene.add_instance("triangle1", instance)
        
        return scene

    @pytest.fixture
    def setup_complex_scene(self):
        """Setup a scene with multiple objects for intersection tests"""
        scene = Scene()
        
        # Create two triangle models at different depths
        model = Model()
        vertices = np.array([
            [0.0, 1.0, 0.0],
            [-1.0, -1.0, 0.0],
            [1.0, -1.0, 0.0]
        ], dtype=np.float32)
        triangles = np.array([[0, 1, 2]], dtype=np.int32)
        model.add_group("default", vertices, triangles)
        scene.add_model("triangle", model)
        
        # Add two instances at different depths
        front = Instance(model="triangle")
        front.set_translation(0.0, 0.0, -5.0)
        scene.add_instance("front", front)
        
        back = Instance(model="triangle")
        back.set_translation(0.0, 0.0, -10.0)
        scene.add_instance("back", back)
        
        return scene

    def test_basic_hit(self, setup_renderer, setup_basic_scene):
        """Test basic ray intersection with a triangle"""
        renderer = setup_renderer
        scene = setup_basic_scene
        
        # Cast ray through center of screen (should hit triangle)
        result = renderer.cast_ray(400, 300, scene)
        assert result is not None
        assert result[0] == "triangle1"
        assert result[1] == "default"

    def test_miss(self, setup_renderer, setup_basic_scene):
        """Test ray missing all geometry"""
        renderer = setup_renderer
        scene = setup_basic_scene
        
        # Cast ray far to the side (should miss)
        result = renderer.cast_ray(0, 0, scene)
        assert result is None

    def test_closest_hit(self, setup_renderer, setup_complex_scene):
        """Test that closest intersection is returned with multiple objects"""
        renderer = setup_renderer
        scene = setup_complex_scene
        
        # Cast ray through center, should hit front triangle
        result = renderer.cast_ray(400, 300, scene)
        assert result is not None
        assert result[0] == "front"  # Should hit front triangle
        assert result[1] == "default"

    def test_backface_culling(self, setup_renderer, setup_basic_scene):
        """Test that backfaces are properly culled"""
        renderer = setup_renderer
        scene = setup_basic_scene
        
        # Rotate triangle 180 degrees to face away from camera
        instance = scene.get_instance("triangle1")
        instance.set_rotation(0.0, np.pi, 0.0)
        
        # Cast ray through center (should miss due to backface culling)
        result = renderer.cast_ray(400, 300, scene)
        assert result is None

    def test_parallel_ray(self, setup_renderer, setup_basic_scene):
        """Test handling of rays parallel to triangle"""
        renderer = setup_renderer
        scene = setup_basic_scene
        
        # Rotate triangle to be edge-on to camera
        instance = scene.get_instance("triangle1")
        instance.set_rotation(np.pi/2, 0.0, 0.0)
        
        # Cast ray through center (should miss due to parallel ray)
        result = renderer.cast_ray(400, 300, scene)
        assert result is None

    def test_edge_hit(self, setup_renderer, setup_basic_scene):
        """Test ray intersection near triangle edge"""
        renderer = setup_renderer
        scene = setup_basic_scene
        
        # Cast ray near edge of triangle
        # Calculate screen position that would hit near edge
        x = 400 + int(0.99 * renderer._canvas.width/4)  # Almost at right edge
        result = renderer.cast_ray(x, 300, scene)
        assert result is not None
        assert result[0] == "triangle1"
        assert result[1] == "default"

    def test_scale_effect(self, setup_renderer, setup_basic_scene):
        """Test ray intersection with scaled geometry"""
        renderer = setup_renderer
        scene = setup_basic_scene
        
        # Scale triangle to be very small
        instance = scene.get_instance("triangle1")
        instance.set_scale(0.1, 0.1, 0.1)
        
        # Cast ray through center (should still hit)
        result = renderer.cast_ray(400, 300, scene)
        assert result is not None
        assert result[0] == "triangle1"
        assert result[1] == "default"
        
        # Scale triangle to be very large
        instance.set_scale(10.0, 10.0, 10.0)
        
        # Cast ray through center (should still hit)
        result = renderer.cast_ray(400, 300, scene)
        assert result is not None
        assert result[0] == "triangle1"
        assert result[1] == "default"
