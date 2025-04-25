==================================================
README - 3DModelingProject_armftn
==================================================

This Computer Vision project focuses on reconstructing a 3D dinosaur model from a sequence of images using techniques like superpixel segmentation, skeleton extraction, and 3D point cloud generation.

Project Contents:
-----------------
1. TP_maillage.m
   - Main script that executes the entire 3D reconstruction pipeline.
   - Tasks: load images, segment into superpixels, extract binary masks, generate object skeletons, reconstruct 3D points, tetrahedralize, filter, and extract the final mesh.
   - Usage: Run this script step-by-step in MATLAB to visualize each processing stage.

2. kmeans_segmentation.m
   - Script that segments images into superpixels using a custom K-means algorithm in the Lab color space.
   - It uses spatial proximity and color similarity to cluster pixels.
   - Usage: Called automatically from TP_maillage.m.

3. extraire_maillage.m
   - Script that extracts the visible external triangular faces from the volumetric tetrahedral mesh.
   - Internal (duplicate) faces are removed to build a clean surface mesh.
   - Usage: Called at the end of TP_maillage.m.

Data Files:
-----------------
- dinoA.png, dinoB.png, dinoL.png, dinoSilhouette.png: Images for visualizations.
- dino_Ps.mat: MATLAB file containing projection matrices for each camera view.
- viff.xy: Contains 2D point correspondences across the multiple views, used for 3D reconstruction.

images Folder:
-----------------
- Contains 36 sequential images (viff.000.ppm to viff.035.ppm) necessary for the dinosaur's 3D reconstruction process.

Additional Notes:
-----------------
- Make sure MATLAB is installed with necessary toolboxes (Image Processing Toolbox recommended).
- The workflow includes segmentation, Voronoï-based skeleton extraction, multi-view triangulation for 3D point cloud reconstruction, tetrahedralization, and final mesh surface extraction.
- Outputs include 3D point clouds and tetrahedral meshes, with an optional export of the final surface.

Enjoy modeling and reconstructing the dinosaur!

armftn
