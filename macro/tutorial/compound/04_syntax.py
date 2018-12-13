#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function
from load import ROOT as R
import constructors as C

number = 1.2345
p0 = C.Points([number])

print(p0)
print(p0.transformations[0])
print(p0.transformations[0].outputs[0])
print(p0.single())

s1 = C.Sum()
s1.add_input('first')
print('Default binding syntax')
s1.sum.inputs[0].connect(p0.points.outputs[0])

s2 = C.Sum()
print('Siplified access')
s2.add_input('first')
s2.single_input().connect(s1.single())

s3 = C.Sum()
s3.add_input('first')
print('Even more simpler way')
s2.sum.sum >> s3.sum.first

s4 = C.Sum()
s4.add_input('first')
print('Even more simpler way')
s3.sum >> s4.sum

s5 = C.Sum()
s5.add_input('first')
print('Even more simpler way')
s4 >> s5

s5 = C.Sum()
s5.add_input('first')
print('The other direction')
s5 << s4

s6 = C.Sum()
s6.add_input('first')
s6.add_input('second')
print('Bind a single output to two inputs')
s5 >> s6.sum.inputs.values()

s7 = C.Sum()
s7.add_input('first')
s7.add_input('second')
print('The other direction')
s6 >> (s6.sum.inputs.first, s6.sum.inputs.second)

result = s7.single().data()[0]

print('Expect:', number*4)
print('Got:   ', result)

