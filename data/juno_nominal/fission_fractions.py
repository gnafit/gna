#!/usr/bin/env python
# encoding: utf-8

# Fission correlation Xubo Ma calculation in DocDB-7413
isotopes    =   [ 'U235', 'U238', 'Pu239', 'Pu241' ]
relsigma       =   [ 0.71*percent, 4.2*percent,  2.1*percent, 3.5*percent ]
correlation = [ [  1.00, -0.22,  -0.53,   -0.18 ],
                [ -0.22,  1.00,   0.18,    0.26 ],
                [ -0.53,  0.18,   1.00,    0.49 ],
                [ -0.18,  0.26,   0.49,    1.00 ] ]
