2d integration
''''''''''''''


..
  .. literalinclude:: ../../../macro/tutorial/complex/02_integral1d_again.py
      :linenos:
      :lines: 4-
      :emphasize-lines: 31,34,35,44,107-119
      :caption: :download:`02_integral1d_again.py <../../../macro/tutorial/complex/02_integral1d_again.py>`

  .. figure:: ../../img/tutorial/02_integral1d_again_graph.png
      :align: center

      Computatinal graph used to integrate function :math:`f(x)=a\sin(x)+b\cos(kx)`. The bin edges are passed via input.

  .. figure:: ../../img/tutorial/02_integral1d_again_1.png
      :align: center

      The Gauss-Legendre quadrature application to the function :eq:`integral_1d_function`. Variable bin width and
      different integration orders per bin.
