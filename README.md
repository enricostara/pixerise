# pixerise

A pure 3D renderer written in Python without making use of existing rendering libraries as OpenGL or Vulkan.

---

Features:
- No dependency to any graphics API like OpenGL or Vulkan
- No dependency to any multimedia library like Pygame or Pyglet
- The library is written in pure Python using NumPy for calculations and Numba for JIT compilation
- Rendering is done by writing pixel data into a buffer that you can display using any graphics system or library you prefer
- Examples are provided to show how to use the library in a simple way by using Pygame to display the result
