from __future__ import print_function
from genomic_neuralnet.util import require_true, RequiredValueNotTrueException 

class Class(object):
    def __init__(self, attribute=None):
        self.attribute = attribute 

    @require_true(lambda self: not self.attribute is None)
    def print_attribute(self):
        print(self.attribute)

def test_require_defined_raise_if_not_defined():
    c = Class(attribute = None)
    try:
        c.print_attribute()
    except RequiredValueNotTrueException:
        return 
    assert False

def test_require_defined_not_raise_if_defined():
    c = Class(attribute = 0)
    try:
        c.print_attribute()
    except RequiredValueNotTrueException:
        assert False 


