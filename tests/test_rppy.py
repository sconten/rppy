#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
test_rppy
----------------------------------

Tests for `rppy` module.
"""

import unittest
import numpy
from rppy import rppy


def test_voight_bound():
    K = numpy.array([36, 2.2])
    f = numpy.array([0.6, 0.4])
    
    assert numpy.abs(rppy.voight_bound(K, f)) < 0.01
    
def test_reuss_bound():
    K = numpy.array([36, 2.2])
    f = numpy.array([0.6, 0.4])
    
    assert numpy.abs(rppy.reuss_bound(K, f) - 5.04) < 0.01

class TestRppy(unittest.TestCase):

    def setUp(self):
        pass

    def test_something(self):
        pass

    def tearDown(self):
        pass

if __name__ == '__main__':
    unittest.main()
