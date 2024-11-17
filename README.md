# pixerise

An easy to use 3D renderer in Python.

#### Features:
- Pixerise is written in pure Python using NumPy and Numba to boosting performance
- Rendering is performed by writing pixel data into a buffer that can be displayed using any system or media library
- No direct dependency on media libraries like Pygame or Pyglet
- No dependency on rendering libraries like OpenGL or Vulkan
- Examples are provided to show how to easily use the library and use Pygame to view the results

#### Who is this for?
Anyone who wants:
- an agnostic 3D renderer that can be integrated with any system or media library in Python
- to do 3D rendering on any device or controller without requiring a GPU
- to extend it to create other 3D libraries or engines that don't require a GPU to run
- to learn 3D rendering from scratch by reading the source code in pure Python

#### Who is this NOT for?
- Anyone who wants to do 3D rendering that takes advantage of GPU acceleration
- Anyone who needs a 3D engine with a complete set of tools to create 3D applications or games