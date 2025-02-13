"""
Test suite for the Scene management module.
Tests Scene class and related classes (Model, Instance, Camera, DirectionalLight).
"""

import numpy as np
from scene import Scene, Model, Instance, Camera, DirectionalLight, ModelInnerGroup


def test_model_inner_group():
    """Test ModelInnerGroup creation and dictionary conversion."""
    # Test creation with vertices and triangles
    vertices = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=np.float32)
    triangles = np.array([[0, 1, 2]], dtype=np.int32)
    group = ModelInnerGroup(vertices=vertices, triangles=triangles)

    assert np.array_equal(group.vertices, vertices)
    assert np.array_equal(group.triangles, triangles)
    assert group.vertex_normals is None

    # Test creation with vertex normals
    normals = np.array([[0, 0, 1], [0, 0, 1], [0, 0, 1]], dtype=np.float32)
    group_with_normals = ModelInnerGroup(
        vertices=vertices, triangles=triangles, vertex_normals=normals
    )
    assert np.array_equal(group_with_normals.vertex_normals, normals)

    # Test from_dict
    data = {
        "vertices": vertices.tolist(),
        "triangles": triangles.tolist(),
        "vertex_normals": normals.tolist(),
    }
    group_from_dict = ModelInnerGroup.from_dict(data)
    assert np.array_equal(group_from_dict.vertices, vertices)
    assert np.array_equal(group_from_dict.triangles, triangles)
    assert np.array_equal(group_from_dict.vertex_normals, normals)


def test_model():
    """Test Model creation and manipulation."""
    model = Model()

    # Test adding a group
    vertices = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=np.float32)
    triangles = np.array([[0, 1, 2]], dtype=np.int32)
    model.add_group("test_group", vertices, triangles)

    assert "test_group" in model.groups
    assert np.array_equal(model.groups["test_group"].vertices, vertices)
    assert np.array_equal(model.groups["test_group"].triangles, triangles)

    # Test to_dict with single default group
    model = Model()
    model.add_group("default", vertices, triangles)
    data = model.to_dict()
    assert "vertices" in data  # Flat structure for default group
    assert "triangles" in data

    # Test to_dict with multiple groups
    model = Model()
    model.add_group("group1", vertices, triangles)
    model.add_group("group2", vertices, triangles)
    data = model.to_dict()
    assert "groups" in data  # Grouped structure for multiple groups
    assert "group1" in data["groups"]
    assert "group2" in data["groups"]

    # Test from_dict with flat structure
    flat_data = {"vertices": vertices.tolist(), "triangles": triangles.tolist()}
    model = Model.from_dict(flat_data)
    assert "default" in model.groups

    # Test from_dict with groups structure
    grouped_data = {
        "groups": {
            "group1": {"vertices": vertices.tolist(), "triangles": triangles.tolist()},
            "group2": {"vertices": vertices.tolist(), "triangles": triangles.tolist()},
        }
    }
    model = Model.from_dict(grouped_data)
    assert "group1" in model.groups
    assert "group2" in model.groups


def test_instance():
    """Test Instance creation and manipulation."""
    # Test default creation
    instance = Instance(model="test_model")
    assert instance.model == "test_model"
    assert np.array_equal(instance.translation, np.zeros(3))
    assert np.array_equal(instance.rotation, np.zeros(3))
    assert np.array_equal(instance.scale, np.ones(3))
    assert np.array_equal(
        instance.color, np.array([200, 200, 200], dtype=np.int32)
    )  # Default gray color

    # Test setters
    instance.set_translation(1, 2, 3)
    instance.set_rotation(np.pi / 2, 0, np.pi / 4)
    instance.set_scale(2, 2, 2)
    instance.color = [128, 179, 255]  # Light blue color as RGB integers

    assert np.array_equal(instance.translation, np.array([1, 2, 3], dtype=np.float32))
    assert np.array_equal(
        instance.rotation, np.array([np.pi / 2, 0, np.pi / 4], dtype=np.float32)
    )
    assert np.array_equal(instance.scale, np.array([2, 2, 2], dtype=np.float32))
    assert np.array_equal(instance.color, np.array([128, 179, 255], dtype=np.int32))

    # Test to_dict and from_dict
    data = instance.to_dict()
    assert data["model"] == "test_model"
    assert data["transform"]["translation"] == [1, 2, 3]
    assert np.allclose(data["transform"]["rotation"], [np.pi / 2, 0, np.pi / 4])
    assert data["transform"]["scale"] == [2, 2, 2]
    assert data["color"] == [128, 179, 255]  # Color as RGB integers

    new_instance = Instance.from_dict(data)
    assert new_instance.model == instance.model
    assert np.array_equal(new_instance.translation, instance.translation)
    assert np.array_equal(new_instance.rotation, instance.rotation)
    assert np.array_equal(new_instance.scale, instance.scale)
    assert np.array_equal(new_instance.color, instance.color)


def test_camera():
    """Test Camera initialization and serialization."""
    # Test initialization
    camera = Camera()
    assert np.allclose(camera.translation, np.zeros(3, dtype=np.float32))
    assert np.allclose(camera.rotation, np.zeros(3, dtype=np.float32))

    # Test property setters
    camera.translation = [1, 2, 3]
    camera.rotation = [0.1, 0.2, 0.3]
    assert np.allclose(camera.translation, np.array([1, 2, 3], dtype=np.float32))
    assert np.allclose(camera.rotation, np.array([0.1, 0.2, 0.3], dtype=np.float32))

    # Test set methods
    camera.set_translation(4, 5, 6)
    camera.set_rotation(0.4, 0.5, 0.6)
    assert np.allclose(camera.translation, np.array([4, 5, 6], dtype=np.float32))
    assert np.allclose(camera.rotation, np.array([0.4, 0.5, 0.6], dtype=np.float32))

    # Test serialization
    camera_data = camera.to_dict()
    assert np.allclose(camera_data["transform"]["translation"], [4, 5, 6])
    assert np.allclose(camera_data["transform"]["rotation"], [0.4, 0.5, 0.6])

    # Test deserialization
    new_camera = Camera.from_dict(camera_data)
    assert np.allclose(new_camera.translation, np.array([4, 5, 6], dtype=np.float32))
    assert np.allclose(new_camera.rotation, np.array([0.4, 0.5, 0.6], dtype=np.float32))


def test_directional_light():
    """Test DirectionalLight initialization and serialization."""
    # Test initialization
    light = DirectionalLight(_direction=np.array([1, 0, 0], dtype=np.float32), _ambient=0.2)
    assert np.array_equal(light.direction, np.array([1, 0, 0], dtype=np.float32))
    assert light.ambient == 0.2

    # Test property setters
    light.direction = [0, 1, 0]
    light.ambient = 0.3
    assert np.array_equal(light.direction, np.array([0, 1, 0], dtype=np.float32))
    assert light.ambient == 0.3

    # Test serialization
    light_data = light.to_dict()
    assert np.array_equal(light_data["direction"], [0, 1, 0])
    assert light_data["ambient"] == 0.3

    # Test deserialization
    new_light = DirectionalLight.from_dict(light_data)
    assert np.array_equal(new_light.direction, np.array([0, 1, 0], dtype=np.float32))
    assert new_light.ambient == 0.3


def test_scene():
    """Test Scene creation and manipulation."""
    scene = Scene()

    # Test default state
    assert len(scene._models) == 0
    assert len(scene._instances) == 0
    assert isinstance(scene._camera, Camera)
    assert isinstance(scene._directional_light, DirectionalLight)

    # Test adding and getting models
    model = Model()
    model.add_group(
        "default",
        np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=np.float32),
        np.array([[0, 1, 2]], dtype=np.int32),
    )
    scene.add_model("test_model", model)

    assert scene.get_model("test_model") == model
    assert scene.get_model("nonexistent") is None

    # Test adding and getting instances
    instance = Instance(model="test_model")
    scene.add_instance("test_instance", instance)

    assert scene.get_instance("test_instance") == instance
    assert scene.get_instance("nonexistent") is None

    # Test setting camera and light
    camera = Camera()
    camera.set_translation(1, 2, 3)
    scene.set_camera(camera)

    light = DirectionalLight(_direction=np.array([-1, -1, -1], dtype=np.float32), _ambient=0.2)
    scene.set_directional_light(light)

    assert scene._camera == camera
    assert scene._directional_light == light

    # Test to_dict and from_dict
    scene_data = {
        "camera": {"transform": {"translation": [1, 2, 3], "rotation": [0, 0, 0]}},
        "models": {
            "cube": {
                "vertices": [[-1, -1, -1], [1, -1, -1], [1, 1, -1]],
                "triangles": [[0, 1, 2]],
            }
        },
        "instances": [
            {
                "name": "cube_instance",
                "model": "cube",
                "transform": {
                    "translation": [0, 0, 5],
                    "rotation": [0, 0, 0],
                    "scale": [1, 1, 1],
                },
                "color": [255, 0, 0],  # Color as RGB integers
            }
        ],
        "lights": {"directional": {"direction": [-1, -1, -1], "ambient": 0.2}},
    }

    scene = Scene.from_dict(scene_data)
    assert "cube" in scene._models
    assert "cube_instance" in scene._instances
    assert np.array_equal(
        scene._camera.translation, np.array([1, 2, 3], dtype=np.float32)
    )
    assert np.array_equal(
        scene._directional_light.direction, np.array([-1, -1, -1], dtype=np.float32)
    )
    assert scene._directional_light.ambient == 0.2

    # Test round-trip serialization
    new_data = scene.to_dict()
    assert "camera" in new_data
    assert "models" in new_data
    assert "instances" in new_data
    assert "directional_light" in new_data
