# nbed - A Python class for processing 4D-STEM nanodiffraction data

nbed is a processing class for exploration and data reduction of nanodiffraction data. nbed helps in 


- aligning diffraction frames
- binning in spatial and diffraction dimensions  
- vectorization by peak detection
- creating virtual images, pseudo Debye-Scherrer and powder patterns
- creating centre of mass maps and orientation maps

Interactive routines have been added for exploration of virtual images and diffraction.
Calibration data base management was added in the last version. 

Currently, nbed supports direct import of PantaRhei .prz files (serialized python object format), EMPAD .raw files
and DECTRIS compressed hdf5 data.   

<bf>
<bf>
    
## Installation

You can use pip to install nbed into your preferred environment.

- First open your preferred shell (Windows users may be using something like gitbash) 

- Activate the Python environment that you wish to use.

- Run

      python -m pip install git+https://github.com/LotharHouben/nbed.git@main



## Test Your Installation

type “python” at the command prompt in your chosen terminal to start a Python session in your active Python environment.

You can now import nbed, create an instance of the nbed class and display the docstring for the LoadFile method:


    ➜ python
    Python 3.11.4 | packaged by conda-forge'
    Type "help", "copyright", "credits" or "license" for more information.
    >>> import nbed
    >>> myset=nbed.pyNBED()
    >>> print(myset.LoadFile.__doc__)

## Documentation

Please refer to the jupyter notebooks with examples in the directory 'examples' at 

    https://github.com/LotharHouben/nbed/examples

The notebooks use a demonstration data set that is available under https://doi.org/10.5281/zenodo.15212905 

Another example is the standalone suite VirtualSTEMExplorerTK in the subdirectory tools.  VirtualSTEMExplorer is a GUI tool to laod DECTRIS HDF5 data files, explore virtual images and diffraction, measure spacings, and export publication ready images.

