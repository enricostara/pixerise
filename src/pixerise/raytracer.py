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

    def _closest_intersection(self, origin: np.ndarray, direction: np.ndarray, 
                            min_t: float, max_t: float) -> tuple[float, dict | None]:
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
        
        return closest_t, closest_sphere

    def _trace_ray(self, origin: np.ndarray, direction: np.ndarray, 
                   min_t: float, max_t: float) -> np.ndarray:
        # Find nearest intersection
        closest_t, closest_sphere = self._closest_intersection(origin, direction, min_t, max_t)
        
        if closest_sphere is None:
            return self._background_color
        
        # Compute intersection point and normal
        point = origin + closest_t * direction
        normal = point - np.array(closest_sphere['center'], dtype=float)
        normal = normal / np.linalg.norm(normal)  # Normalize
        
        # Apply lighting with specular component
        intensity = self._compute_lighting(point, normal, -direction, closest_sphere.get('specular', 50))
        color = np.array(closest_sphere['color'], dtype=float) * intensity
        return np.clip(color, 0, 255).astype(int)

    def _compute_lighting(self, point: np.ndarray, normal: np.ndarray, view_dir: np.ndarray, specular: float) -> float:
        intensity = 0.0
        specular = max(0, specular)  # Ensure non-positive values are treated as 0
        
        for light in self._scene.get('lights', []):
            if light['type'] == 'ambient':
                intensity += light['intensity']
            else:
                if light['type'] == 'point':
                    light_dir = np.array(light['position'], dtype=float) - point
                    light_distance = np.linalg.norm(light_dir)
                    light_dir = light_dir / light_distance  # Normalize
                else:  # directional light
                    light_dir = np.array(light['direction'], dtype=float)
                    light_dir = light_dir / np.linalg.norm(light_dir)  # Normalize
                    light_distance = float('inf')
                
                # Check for shadows by casting a ray from the point to the light
                shadow_t, shadow_sphere = self._closest_intersection(
                    point + normal * 1e-5,  # Offset point slightly to avoid self-intersection
                    light_dir,
                    0.001,  # Minimum distance to avoid self-intersection
                    light_distance  # Maximum distance is the distance to the light
                )
                
                # Skip this light if there's an object blocking it
                if shadow_sphere is not None:
                    continue
                
                # Compute diffuse lighting (Lambert's law)
                n_dot_l = np.dot(normal, light_dir)
                if n_dot_l > 0:
                    intensity += light['intensity'] * n_dot_l
                    
                    # Compute specular lighting only if specular > 0
                    if specular > 0:
                        # Calculate reflection vector: R = 2(N·L)N - L
                        reflect_dir = 2 * n_dot_l * normal - light_dir
                        reflect_dir = reflect_dir / np.linalg.norm(reflect_dir)  # Normalize
                        
                        # Calculate specular component using Phong model
                        r_dot_v = np.dot(reflect_dir, view_dir)
                        if r_dot_v > 0:
                            intensity += light['intensity'] * (r_dot_v ** specular)
        
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
