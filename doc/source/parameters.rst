Parameters
=============

We have already seen at least how to:

- request a parameter when constructing an experiment (``ns.reqparameter(...)``);
- find a parameter and change its value (``ns[...].set(v)`` or
  ``ns --value ... v`` ).

Here we'll try some more examples. First, let's try to make the
``gaussianpeak`` model reuse existing ``Mu`` parameter, that will be
in separate namespace which we'll call ``common``::
  python ./gna -- ns --define common.Mu central=1 sigma=0.01 \
                  --push common
               -- gaussianpeak --name peak1

With the first command (``ns``) we have defined new parameter with
full path ``common.Mu``, central value 1 and absolute uncertainty
0.1 and then activated the ``common`` namespace by using the
``--push`` option. Activation means that each name inside the
``common`` namspace will be available during look ups, in particular
during object binding. Since ``Mu`` parameter will be available during
``gaussianpeak`` execution, the corresponding ``ns.reqparameter``
won't create any new parameter. In other words, there will be no
``peak1.Mu`` as before and ``common.Mu`` will be used by our
observable instead. As a consequence, as we have alreade set its
central (and default) value to 1, if you'll plot the
``peak1.spectrum``, you'll see the peak without any additional ``ns --value``.

In the same manner, we can create several experimental models and make
some parameters shared between them. For example, if we have two
different experiments, that are expecting to observe two different
and independent peaks, but with exactly the same background rate, it
makes sense to make the ``BackgroundRate`` parameter common::
  python ./gna -- ns --define common.BackgroundRate central=1 sigma=0.1 \
                  --push common
               -- gaussianpeak --name peak1
               -- gaussianpeak --name peak2
               -- ns --value peak1.Mu 1 --value peak2.Mu 2
               -- spectrum --plot peak1/spectrum --plot peak2/spectrum
               -- ns --value common.BackgroundRate 2
               -- spectrum --plot peak1/spectrum --plot peak2/spectrum

As you can see, we treat ``Mu`` of both peaks independently, while the
background rate is really common and affects both peaks at the same
time.

