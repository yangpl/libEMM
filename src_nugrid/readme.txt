steps to generate resistivity model on uniform and non-uniform grid:

1. compile code:
   make

2. run code:
   ./main

3. visualize the model by paraview:
   convert binary data into vtk:
   python3 output_model_vtk.py ('pip install pyvtk' if pyvtk has not been installed)
