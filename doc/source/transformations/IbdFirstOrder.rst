IbdFirstOrder
~~~~~~~~~~~~~

Description
^^^^^^^^^^^
Computes IBD cross section in a first order of perturbation theory, `xsec`
transformation, 
transforms :math:`E_{\text{e}} \rightarrow E_{\nu}`, `Enu` transformation, and
computes Jacobian for change of variables :math:`E_{\nu} \rightarrow E_{\nu}(E_{\text{e}})`.

Inputs
^^^^^^

  1. One-dimensional arrays of :math:`E_{\text{e}}` and :math:`\cos\theta` for `Enu` transformation.
  2. One-dimensional arrays of :math:`E_\nu` and :math:`\cos\theta` for `xsec` transformation.
  3. One-dimensional arrays of :math:`E_\nu`, :math:`\cos\theta` and :math:`E_{\text{e}}` for `jacobian` transformation.
  

Outputs
^^^^^^^

1. Two-dimensional array of :math:`E_{\nu}`.
2. Two-dimensional array of :math:`\sigma_{\text{IBD}}`.
3. Two-dimensional array of Jacobian values as function of :math:`E_{\text{e}}` and :math:`\cos \theta`.  

Implementation
^^^^^^^^^^^^^^
See the Daya Bay paper on oscillation analysis based on 1230 days of data.
