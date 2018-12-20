Making parameters via bundles
'''''''''''''''''''''''''''''

.. literalinclude:: ../../../macro/tutorial/bundles/03_bundle_parameters.py
    :linenos:
    :lines: 4-
    :caption: :download:`03_bundle_parameters.py <../../../macro/tutorial/bundles/03_bundle_parameters.py>`

.. code-block:: text

   Variables in namespace 'bundle1.SA.rate0':
     D1                   =          1 │           1±        0.01 [          1%] │ Flux normalization SA->D1
     D2                   =          3 │           3±        0.03 [          1%] │ Flux normalization SA->D2
     D3                   =          5 │           5±        0.05 [          1%] │ Flux normalization SA->D3
   Variables in namespace 'bundle1.SB.rate0':
     D1                   =          2 │           2±        0.02 [          1%] │ Flux normalization SB->D1
     D2                   =          4 │           4±        0.04 [          1%] │ Flux normalization SB->D2
     D3                   =          6 │           6±        0.06 [          1%] │ Flux normalization SB->D3
   Variables in namespace 'bundle2.rate1.SA.D1':
     e0                   =          1 │           1±        0.01 [          1%] │ Flux normalization SA->D1 (e0)
     e1                   =          1 │           1±        0.01 [          1%] │ Flux normalization SA->D1 (e1)
   Variables in namespace 'bundle2.rate1.SA.D2':
     e0                   =          3 │           3±        0.09 [          3%] │ Flux normalization SA->D2 (e0)
     e1                   =          3 │           3±        0.09 [          3%] │ Flux normalization SA->D2 (e1)
   Variables in namespace 'bundle2.rate1.SA.D3':
     e0                   =          5 │           5±        0.25 [          5%] │ Flux normalization SA->D3 (e0)
     e1                   =          5 │           5±        0.25 [          5%] │ Flux normalization SA->D3 (e1)
   Variables in namespace 'bundle2.rate1.SB.D1':
     e0                   =          2 │           2±        0.04 [          2%] │ Flux normalization SB->D1 (e0)
     e1                   =          2 │           2±        0.04 [          2%] │ Flux normalization SB->D1 (e1)
   Variables in namespace 'bundle2.rate1.SB.D2':
     e0                   =          4 │           4±        0.16 [          4%] │ Flux normalization SB->D2 (e0)
     e1                   =          4 │           4±        0.16 [          4%] │ Flux normalization SB->D2 (e1)
   Variables in namespace 'bundle2.rate1.SB.D3':
     e0                   =          6 │           6±        0.36 [          6%] │ Flux normalization SB->D3 (e0)
     e1                   =          6 │           6±        0.36 [          6%] │ Flux normalization SB->D3 (e1)
   Variables in namespace 'bundle3':
     constant             =         -1 │          -1±       -0.04 [          4%] │ some constant






