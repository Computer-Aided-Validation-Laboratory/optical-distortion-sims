# Validation Case

A single camera, window and sample is used for validating Blender against Mitsuba

### Camera
The actual camera used is used in this validation:
- 5328 x 4608 px
- 35 mm focal length
- 400 mm away from the sample

### Window
The window is exactly the same, and is imported as a MOOSE file. 
This mesh needs to be triangulated for import into Mistuba. 
In Blender, a Glass BSDF shader is used, and in Mitsuba a dielectric bsdf is used. 

### Sample
A checkerboard sample is used, with a 16x14 grid. 
This is placed at the location of the actual specimen.


### Rendering 
The images are rendered with 1024 samples per pixels. 

