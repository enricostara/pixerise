from pixerise.canvas import Canvas
from pixerise.viewport import ViewPort
import numpy as np
import math


class RayTracer:

    def __init__(self, canvas: Canvas, viewport: ViewPort, scene: dict, background_color=(32, 32, 32)):
        self._canvas = canvas
        self._viewport = viewport
        self._scene = scene
        self._background_color = np.array(background_color, dtype=int)

    def render(self, origin: (float, float, float)):
        origin = np.array(origin, dtype=float)
        # Trace a ray through each pixel
        for x in range(-self._canvas.half_width, self._canvas.half_width):
            for y in range(-self._canvas.half_height + 1, self._canvas.half_height + 1):
                direction = np.array(self._viewport.canvas_to_viewport_direction(x, y), dtype=float)
                color = self._trace_ray(origin, direction, 1, float('inf'))
                self._canvas.draw_unchecked_point(x, y, tuple(color))

    def _trace_ray(self, origin: np.ndarray, direction: np.ndarray, 
                   min_t: float, max_t: float) -> np.ndarray:
        # Find nearest intersection
        closest_t = float('inf')
        closest_sphere = None
        
        for sphere in self._scene['spheres']:
            t1, t2 = self._intersect_ray_sphere(origin, direction, sphere)
            if min_t <= t1 < max_t and t1 < closest_t:
                closest_t = t1
                closest_sphere = sphere
            if min_t <= t2 < max_t and t2 < closest_t:
                closest_t = t2
                closest_sphere = sphere
        
        if closest_sphere is None:
            return self._background_color
        
        # Compute intersection point and normal
        point = origin + closest_t * direction
        normal = point - np.array(closest_sphere['center'], dtype=float)
        normal = normal / np.linalg.norm(normal)  # Normalize
        
        # Apply lighting
        intensity = self._compute_lighting(point, normal)
        color = np.array(closest_sphere['color'], dtype=float) * intensity
        return np.clip(color, 0, 255).astype(int)

    def _compute_lighting(self, point: np.ndarray, normal: np.ndarray) -> float:
        intensity = 0.0
        
        for light in self._scene.get('lights', []):
            if light['type'] == 'ambient':
                intensity += light['intensity']
            else:
                if light['type'] == 'point':
                    light_dir = np.array(light['position'], dtype=float) - point
                else:  # directional light
                    light_dir = np.array(light['direction'], dtype=float)
                    
                light_dir = light_dir / np.linalg.norm(light_dir)  # Normalize
                # Compute diffuse lighting (Lambert's law)
                n_dot_l = np.dot(normal, light_dir)
                if n_dot_l > 0:
                    intensity += light['intensity'] * n_dot_l
        
        return intensity

    def _intersect_ray_sphere(self, origin: np.ndarray, direction: np.ndarray, 
                             sphere: dict) -> (float, float):
        center = np.array(sphere['center'], dtype=float)
        radius = sphere['radius']
        oc = origin - center
        
        a = np.dot(direction, direction)
        b = 2 * np.dot(oc, direction)
        c = np.dot(oc, oc) - radius * radius
        
        discriminant = b * b - 4 * a * c
        if discriminant < 0:
            return float('inf'), float('inf')
            
        t1 = (-b + np.sqrt(discriminant)) / (2 * a)
        t2 = (-b - np.sqrt(discriminant)) / (2 * a)
        return t1, t2
