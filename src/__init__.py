# classes

# header hooks

_PROGRAM_NAME = 'nbed'
_USER_NAME = ''
def set_author(author):
    """ Accepts a string, which will be written to the "authoring_user" field in any EMD file headers
        written during this Python session
    """
    global _USER_NAME
    _USER_NAME = author

from .helpers import (
    ParabolaFit2D,   
    convolve2D,
    read_empad,
    bytscl
)

# classes                                                                                             

from nbed.classes import (
    pyNBED
)


# header hooks                                                                                        
