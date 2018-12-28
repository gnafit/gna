Replicating graphs
''''''''''''''''''

Now let us learn the second capability of the bundles: graph generation and replication. We will use the example of the
:ref:`energy resolution transformation<EnergyResolution>`. In short, the energy resolution transformation smears the
input histogram bin by bin. Each bin is smeared with a Gaussian. A width is defined based on the bin center by a
formula, that depends on three parameters: :math:`a`, :math:`b` and :math:`c`.

The energy resolution object contains 2+ transformations: `matrix` transformation computes the smearing matrix based on
the values of the parameters, at least one `smear` transformation smears the input histogram with smearing matrix.

We now will define a bundle which:

#. Defines the parameters for :math:`a`, :math:`b` and :math:`c`.

   + If major index is specified, different parameters are defined for each iteration.
   + Minor indices are ignored.

#. Defines the energy resolution object for each major iteration.

   + The `matrix` transformation depends on the current parameters :math:`a`, :math:`b` and :math:`c`.
     The bundle provides an input of the `matrix` transformation for each major iteration. The bin edges output should
     be connected to it.
   + New `smear` transformation is added for each minor iteration.
   + The bundle provides an input/output pair on each minor+major iteration.

#. Optional features:

   + Label formats.
   + Merge transformations. Do not create a new transformation for each minor index. Use the same transformation to
     process all the inputs. The procedure is explained :ref:`here <tutorial_topology>`.

Energy resolution bundle
++++++++++++++++++++++++

.. literalinclude:: ../../../macro/tutorial/bundles/detector_eres_ex01.py
    :linenos:
    :lines: 4-
    :caption: :download:`detector_eres_ex01.py <../../../macro/tutorial/bundles/detector_eres_ex01.py>`

.. literalinclude:: ../../../macro/tutorial/bundles/05_bundle_eres.py
    :linenos:
    :lines: 4-
    :caption: :download:`05_bundle_eres.py <../../../macro/tutorial/bundles/05_bundle_eres.py>`

.. figure:: ../../ img/tutorial/05_bundle_eres_graph.png
    :align: center

.. figure:: ../../ img/tutorial/05_bundle_eres.png
    :align: center

Energy resolution replicated
++++++++++++++++++++++++++++

.. literalinclude:: ../../../macro/tutorial/bundles/detector_eres_ex02.py
    :linenos:
    :lines: 4-
    :caption: :download:`detector_eres_ex02.py <../../../macro/tutorial/bundles/detector_eres_ex02.py>`

.. literalinclude:: ../../../macro/tutorial/bundles/05_bundle_eres_upd.py
    :linenos:
    :lines: 4-
    :caption: :download:`05_bundle_eres_upd.py <../../../macro/tutorial/bundles/05_bundle_eres_upd.py>`

.. figure:: ../../ img/tutorial/05_bundle_eres_upd_graph0.png
    :align: center

.. figure:: ../../ img/tutorial/05_bundle_eres_upd_graph1.png
    :align: center

.. figure:: ../../ img/tutorial/05_bundle_eres_upd.png
    :align: center

Energy resolution replicated (and merged)
+++++++++++++++++++++++++++++++++++++++++

.. figure:: ../../ img/tutorial/05_bundle_eres_upd_merged_graph0.png
    :align: center

.. figure:: ../../ img/tutorial/05_bundle_eres_upd_merged_graph1.png
    :align: center
