Subsubsection name
^^^^^^^^^^^^^^^^^^

.. code-block:: python

    indices = dict()
    incices['b'] = [ 'acc', 'lihe', 'fastn', 'amc', 'alphan' ]
    indices['c'] = [ 'comp'+str(i) for i in range (1, 4) ]
    indices['i'] = [ 'U5', 'U8', 'Pu9', 'Pu1' ]
    indices['r'] = [ 'DB'+str(i) for i in range(1,3)]+['LA'+str(i) for i in range(1,5) ]
    indices['d'] = [ 'AD'+str(i) for i in [ 11, 12, 21, 22, 31, 32, 33, 34 ] ]

    # expr is a string to be evaluated within special environment
    expr = bkg[b] \
           + oscprob_w[c] * rebin[d] | eres[d] | lsnl[d] | iav[d] | \
             eff[d] * lt[d] * mass[d] \
           * sum[r] | dist_weight[d,r] \
           * sum[i] | \
             int2d | \
               csc(enu(), cost()) * jacobian(enu(), cost()) * (oscprob_item0 + oscprob_item[c,d,r]) \
               * ( ff[i,r] * power[r] * spec_n() * spec_corr() * spec_uncorr[i]() * spec[i](enu()) \
                 + ff[i,r] * power[r] * offeq_norm[i, r] * spec_offeq[i,r](enu()) \
                 + power[r] * snf_norm[r] * snf[r](enu()) )

    cfg = NestedDict(
        bundle = 'expression',
        expression = res,
        indices = indices,
        
        oscprob = NestedDict( bundle='...', ... ),
        rebin   = NestedDict( bundle='rebin', ... ),
        int2d   = NestedDict( bundle='gh2d', edges=..., provides=['enu'] ),
        spec    = NestedDict( bundle='spectrum', ..., )
    )

Legend:
    - Variables:
         * :code:`name` -- variable without indices. Located within namespaces.
         * :code:`name[i,j,...]` -- variable with indices. Located within namespaces.
    
    - Functions, located within shared environment:
         * :code:`name()` -- function without indices and arguments. Single output. 
         * :code:`name(...)` -- function without indices. With arguments. Has inputs and output.
         * :code:`name[i,j...](...)` -- function with indices. With arguments. Has inputs and output.
         * :code:`name[i,j...] | ...` -- function, equivalent to :code:`name[i,j...](...)`.
            + '|' is implemented by string substitution: '|' is replaced with '(' and ')' is added to the end of the
              string.
     
    - Special functions and syntax:
        * :code:`fun1()*fun2()*...` produces Product.
        * :code:`fun1()+fun2()+...` produces Sum.
        * :code:`var1*fun1()+var2*fun2()+...` produces WeightedSum.
        * :code:`sum[i,j](...)` -- Sums indices in `i,j` output carries all remaining indices.
        * Function call means connection of the output to the function input.
    
    - Bundles:
        * Each name is provided by some bundle. 'provides' field tells which bundle to load to provide function or
          variable.

    - Simplification:
        * In the first implementation the simplification may be omitted. The formula structure is setup manually.
        * In more complex version, one may allow to:
            + expand sums like  :code:`fun1() * (fun2() + fun3())`.
            + move functions without indices to the left.
            + move variables without indices to the left.
            + move variables with indices to the left up to first limiting sum.








