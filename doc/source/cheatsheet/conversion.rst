Types conversion
^^^^^^^^^^^^^^^^

The conversion is done by means of ``converters`` module:

.. code-block:: python

   from converters import convert

Function ``convert(object, target_type)`` converts an object (usually an array) to the ``target_type``.
The following types are recognized:
    + ``ROOT.vector`` or ``'stdvector'``
    + ``ROOT.Points`` or ``'points'``
    + ``ROOT.Eigen.MatrixXd`` or ``'eigenmatrix'``
    + ``ROOT.Eigen.VectorXd`` or ``'eigenvector'``
    + ``ROOT.Eigen.ArrayXd`` or ``'eigenarray'``
    + ``ROOT.Eigen.ArrayXXd`` or ``'eigenarray2d'``
    + ``numpy.ndarray`` or ``'array'``
    + ``numpy.matrixlib.defmatrix.matrix`` or ``'matrix'``

STD vector
""""""""""

.. table::  **STD vector** :math:`\leftrightarrow` **numpy**
   :widths: 100 80

   +-----------------------------------------+------------------------------------------------------------+
   | ``convert(stdvec, numpy.ndarray)``      | any numerical std::vector to numpy array (type is guessed) |
   +-----------------------------------------+------------------------------------------------------------+
   | ``convert(stdvec, 'array')``            | same as above                                              |
   +-----------------------------------------+------------------------------------------------------------+
   | ``convert(stdvec, 'array', dtype='i')`` | same as above with explicit data type (see numpy dtype)    |
   +-----------------------------------------+------------------------------------------------------------+
   | ``convert(array, ROOT.vector)``         | numpy array to std::vector (type is guessed)               |
   +-----------------------------------------+------------------------------------------------------------+
   | ``convert(array, 'stdvector')``         | same as above                                              |
   +-----------------------------------------+------------------------------------------------------------+

Eigen
"""""

.. table:: **numpy** :math:`\rightarrow` **Eigen**
   :widths: 100 80

   +-------------------------------------------+----------------------------------------------------------+
   | ``convert(array1d, ROOT.Eigen.ArrayXd)``  | 1d numpy array to Eigen Array (double)                   |
   +-------------------------------------------+----------------------------------------------------------+
   | ``convert(array1d, 'eigenarray')``        | same as above                                            |
   +-------------------------------------------+----------------------------------------------------------+
   | ``convert(array2d, ROOT.Eigen.ArrayXXd)`` | 2d numpy array to Eigen 2D Array (double)                |
   +-------------------------------------------+----------------------------------------------------------+
   | ``convert(array2d, 'eigenarray2d')``      | same as above                                            |
   +-------------------------------------------+----------------------------------------------------------+
   | ``convert(vector, ROOT.Eigen.VectorXd)``  | 1d numpy vector (matrix column) to Eigen Vector (double) |
   +-------------------------------------------+----------------------------------------------------------+
   | ``convert(vector, 'eigenvector')``        | same as above                                            |
   +-------------------------------------------+----------------------------------------------------------+
   | ``convert(matrix, ROOT.Eigen.MatrixXd)``  | 2d numpy array/matrix to Eigen Matrix (double)           |
   +-------------------------------------------+----------------------------------------------------------+
   | ``convert(matrix, 'eigenmatrix')``        | same as above                                            |
   +-------------------------------------------+----------------------------------------------------------+

|

.. table:: **Eigen** :math:`\rightarrow` **numpy**
   :widths: 100 80

   +------------------------------------+-----------------------------------------------+
   | ``convert(eigenarray, 'array')``   | 1d Eigen Array to numpy array                 |
   +------------------------------------+-----------------------------------------------+
   | ``convert(eigenarray2d, 'array')`` | 2d Eigen Array to numpy array (same as above) |
   +------------------------------------+-----------------------------------------------+
   | ``convert(eigenmatrix, 'array')``  | Eigen Matrix to numpy array                   |
   +------------------------------------+-----------------------------------------------+
   | ``convert(eigenmatrix, 'matrix')`` | Eigen Matrix to numpy matrix                  |
   +------------------------------------+-----------------------------------------------+
   | ``convert(eigenvector, 'matrix')`` | Eigen Vector to numpy matrix                  |
   +------------------------------------+-----------------------------------------------+

GNA types
"""""""""

.. table:: **numpy** :math:`\rightarrow` **GNA types**
   :widths: 100 80

   +---------------------------------+-----------------------------+
   | ``convert(array, ROOT.Points)`` | numpy 1d/2d array to Points |
   +---------------------------------+-----------------------------+
   | ``convert(array, 'points')``    | same as above               |
   +---------------------------------+-----------------------------+
