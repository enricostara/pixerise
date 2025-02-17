"""
Test suite for the Scene management module.
Tests Scene class and related classes (Model, Instance, Camera, DirectionalLight).
"""

import numpy as np
from scene import Scene, Model, Instance, Camera, DirectionalLight


def test_model_inner_group():
    """Test Model.Group initialization and serialization."""
    # Test initialization
    vertices = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=np.float32)
    triangles = np.array([[0, 1, 2]], dtype=np.int32)
    vertex_normals = np.array([[0, 0, 1], [0, 0, 1], [0, 0, 1]], dtype=np.float32)

    group = Model.Group(
        _vertices=vertices, _triangles=triangles, _vertex_normals=vertex_normals
    )

    # Test read-only properties
    assert np.array_equal(group.vertices, vertices)
    assert np.array_equal(group.triangles, triangles)
    assert np.array_equal(group.vertex_normals, vertex_normals)

    # Test properties return copies
    vertices_copy = group.vertices
    triangles_copy = group.triangles
    vertex_normals_copy = group.vertex_normals

    # Modifying copies shouldn't affect original
    vertices_copy[0, 0] = 999
    triangles_copy[0, 0] = 999
    vertex_normals_copy[0, 0] = 999

    assert not np.array_equal(vertices_copy, group.vertices)
    assert not np.array_equal(triangles_copy, group.triangles)
    assert not np.array_equal(vertex_normals_copy, group.vertex_normals)

    # Test without vertex normals
    group_no_normals = Model.Group(_vertices=vertices, _triangles=triangles)
    assert group_no_normals.vertex_normals is None

    # Test serialization through Model
    model = Model()
    model.add_group(vertices, triangles, vertex_normals, "test")

    # Test model serialization
    model_dict = model.to_dict()
    assert "groups" in model_dict
    assert "test" in model_dict["groups"]
    assert "vertices" in model_dict["groups"]["test"]
    assert "triangles" in model_dict["groups"]["test"]
    assert "vertex_normals" in model_dict["groups"]["test"]

    # Test model deserialization
    model2 = Model.from_dict(model_dict)
    assert "test" in model2._groups
    group2 = model2._groups["test"]
    assert np.array_equal(group2.vertices, vertices)
    assert np.array_equal(group2.triangles, triangles)
    assert np.array_equal(group2.vertex_normals, vertex_normals)


def test_model():
    """Test Model initialization and serialization."""
    # Test initialization
    model = Model()
    assert len(model.groups) == 0  # Use the property instead of _groups

    # Test adding groups
    vertices = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=np.float32)
    triangles = np.array([[0, 1, 2]], dtype=np.int32)
    vertex_normals = np.array([[0, 0, 1], [0, 0, 1], [0, 0, 1]], dtype=np.float32)

    model.add_group(vertices, triangles, vertex_normals, "test_group")
    assert "test_group" in model.groups
    assert len(model.groups) == 1

    group = model.groups["test_group"]
    assert np.array_equal(group.vertices, vertices)
    assert np.array_equal(group.triangles, triangles)
    assert np.array_equal(group.vertex_normals, vertex_normals)

    # Test serialization
    model_dict = model.to_dict()
    assert "groups" in model_dict
    assert "test_group" in model_dict["groups"]
    assert "vertices" in model_dict["groups"]["test_group"]
    assert "triangles" in model_dict["groups"]["test_group"]
    assert "vertex_normals" in model_dict["groups"]["test_group"]

    # Test deserialization
    model2 = Model.from_dict(model_dict)
    assert "test_group" in model2._groups
    group2 = model2._groups["test_group"]
    assert np.array_equal(group2.vertices, vertices)
    assert np.array_equal(group2.triangles, triangles)
    assert np.array_equal(group2.vertex_normals, vertex_normals)

    # Test groups property is read-only (returns a copy)
    original_groups = model.groups
    group = original_groups["test_group"]
    original_groups["new_group"] = group  # This shouldn't affect the model
    assert "new_group" not in model.groups

    # Test single default group serialization
    default_model = Model()
    default_model.add_group(vertices, triangles, vertex_normals)  # Use default name
    assert "default" in default_model.groups
    assert len(default_model.groups) == 1


def test_instance():
    """Test Instance initialization and serialization."""
    # Test initialization
    instance = Instance(_model="test_model")
    assert instance.model == "test_model"
    assert np.allclose(instance.translation, np.zeros(3, dtype=np.float32))
    assert np.allclose(instance.rotation, np.zeros(3, dtype=np.float32))
    assert np.allclose(instance.scale, np.ones(3, dtype=np.float32))
    assert np.array_equal(instance.color, np.array([200, 200, 200], dtype=np.int32))

    # Test setters
    instance.translation = np.array([1, 2, 3], dtype=np.float32)
    instance.rotation = np.array([0.1, 0.2, 0.3], dtype=np.float32)
    instance.scale = np.array([2, 2, 2], dtype=np.float32)
    instance.color = np.array([255, 128, 0], dtype=np.int32)

    assert np.allclose(instance.translation, np.array([1, 2, 3], dtype=np.float32))
    assert np.allclose(instance.rotation, np.array([0.1, 0.2, 0.3], dtype=np.float32))
    assert np.allclose(instance.scale, np.array([2, 2, 2], dtype=np.float32))
    assert np.array_equal(instance.color, np.array([255, 128, 0], dtype=np.int32))

    # Test group colors and visibility
    group_color = np.array([100, 150, 200], dtype=np.int32)
    instance.set_group_color("group1", group_color)

    # Test color is returned as is (no copy)
    retrieved_color = instance.get_group_color("group1")
    assert retrieved_color is instance._groups["group1"].color
    assert np.array_equal(retrieved_color, group_color)

    # Test non-existent group returns None for color and True for visibility
    assert instance.get_group_color("non_existent") is None
    assert instance.get_group_visibility("non_existent") is True

    # Test setting visibility without color
    instance.set_group_visibility("group2", False)
    assert instance.get_group_color("group2") is None
    assert instance.get_group_visibility("group2") is False

    # Test setting color to None preserves visibility
    instance.set_group_visibility("group3", False)
    instance.set_group_color("group3", group_color)
    assert np.array_equal(instance.get_group_color("group3"), group_color)
    assert instance.get_group_visibility("group3") is False
    instance.set_group_color("group3", None)
    assert instance.get_group_color("group3") is None
    assert instance.get_group_visibility("group3") is False

    # Test GroupState class directly
    state = Instance.Group(color=group_color, visible=False)
    assert np.array_equal(state.color, group_color)
    assert state.visible is False

    # Test serialization
    instance_dict = instance.to_dict()
    assert instance_dict["model"] == "test_model"
    assert "transform" in instance_dict
    assert "translation" in instance_dict["transform"]
    assert "rotation" in instance_dict["transform"]
    assert "scale" in instance_dict["transform"]
    assert "color" in instance_dict
    assert "groups" in instance_dict

    # Test deserialization
    instance2 = Instance.from_dict(instance_dict)
    assert instance2.model == "test_model"
    assert np.allclose(instance2.translation, instance.translation)
    assert np.allclose(instance2.rotation, instance.rotation)
    assert np.allclose(instance2.scale, instance.scale)
    assert np.array_equal(instance2.color, instance.color)


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
    light = DirectionalLight(
        _direction=np.array([1, 0, 0], dtype=np.float32), _ambient=0.2
    )
    assert np.array_equal(light.direction, np.array([1, 0, 0], dtype=np.float32))
    assert light.ambient == 0.2

    # Test property setters
    light.direction = np.array([0, 1, 0], dtype=np.float32)
    light.ambient = 0.5
    assert np.array_equal(light.direction, np.array([0, 1, 0], dtype=np.float32))
    assert light.ambient == 0.5

    # Test serialization
    light_data = light.to_dict()
    assert np.array_equal(light_data["direction"], [0, 1, 0])
    assert light_data["ambient"] == 0.5

    # Test deserialization
    new_light = DirectionalLight.from_dict(light_data)
    assert np.array_equal(new_light.direction, np.array([0, 1, 0], dtype=np.float32))
    assert new_light.ambient == 0.5


def test_scene():
    """Test Scene initialization and manipulation."""
    scene = Scene()

    # Test adding and getting models
    model = Model()
    scene.add_model("test_model", model)

    assert scene.get_model("test_model") == model
    assert scene.get_model("nonexistent") is None

    # Test adding and getting instances
    instance = Instance(_model="test_model")
    scene.add_instance("test_instance", instance)

    assert scene.get_instance("test_instance") == instance
    assert scene.get_instance("nonexistent") is None

    # Test setting camera and light
    camera = Camera()
    camera.set_translation(1, 2, 3)
    scene.set_camera(camera)

    light = DirectionalLight(
        _direction=np.array([-1, -1, -1], dtype=np.float32), _ambient=0.2
    )
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
                "groups": {
                    "group1": {
                        "color": [100, 150, 200],
                        "visible": True,
                    },
                    "group2": {
                        "color": None,
                        "visible": False,
                    },
                },
            }
        ],
        "lights": {"directional": {"direction": [-1, -1, -1], "ambient": 0.2}},
    }

    scene = Scene.from_dict(scene_data)
    assert "cube" in scene._models
    assert "cube_instance" in scene._instances

    # Verify instance groups after deserialization
    deserialized_instance = scene.get_instance("cube_instance")
    assert np.array_equal(
        deserialized_instance.get_group_color("group1"),
        np.array([100, 150, 200], dtype=np.int32),
    )
    assert deserialized_instance.get_group_visibility("group1") is True
    assert deserialized_instance.get_group_color("group2") is None
    assert deserialized_instance.get_group_visibility("group2") is False

    assert np.allclose(scene._camera.translation, np.array([1, 2, 3], dtype=np.float32))
    assert np.allclose(
        scene._directional_light.direction, np.array([-1, -1, -1], dtype=np.float32)
    )
    assert scene._directional_light.ambient == 0.2

    # Test round-trip serialization
    new_data = scene.to_dict()
    assert "camera" in new_data
    assert "models" in new_data
    assert "instances" in new_data
    assert "directional_light" in new_data
