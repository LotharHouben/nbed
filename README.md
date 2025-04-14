# nbed - A Python class for processing 4D-STEM nanodiffraction data

nbed is a processing class for data reduction of nanodiffraction data. nbed helps in 


- aligning diffraction frames
- vectorization by peak detection
- creating virtual images, pseudo Debye-Scherrer and powder patterns

Currently, nbed supports direct import of PantaRhei .prz files (serialized python object format) and EMPAD .raw files. 

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

You can now import nbed, create an instance of the nbed class and dosplay the docstring for the LoadFile method:


    ➜ python
    Python 3.11.4 | packaged by conda-forge'
    Type "help", "copyright", "credits" or "license" for more information.
    >>> import nbed
    >>> myset=nbed.pyNBED()
    >>> print(myset.LoadFile.__doc__)

## Documentation

Please refer to the jupyter notebooks with examples in the directory 'examples at 

    https://github.com/LotharHouben/nbed 
