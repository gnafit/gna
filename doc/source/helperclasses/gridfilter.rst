GridFilter
~~~~~~~~~~

Description
^^^^^^^^^^^

Contais methods for working with cross-sections of the predetermined function. Function is represented as a matrix with values computed with the given grid.

The following operations can be done:

* Get cross-section :math:`z=value` with the given function value;

* Get extended cross-section with the given function value and deviation from the original cross-section contour. Extended cross-section contains :math:`|z-value|  ≤  deviation` points;

* Get extended cross-section with the given function value where deviation is computed automaticly and depends on the gradient vector length at cross-section points.

Constructors
^^^^^^^^^^^^


``GridFilter(const Eigen::MatrixXd & mat, double xStep, double yStep, int initDeviation, double gradInfl, double allowErr)``

``GridFilter(int rows, int columns, double* mat, double xStep, double yStep, int initDeviation, double gradInfl, double allowErr)``

where 

* :math:`mat` is matrix contains float values;

* :math:`rows` and :math:`columns` are sizes of matrix :math:`mat`;

* :math:`xStep` and :math:`yStep` are grid steps;

* :math:`initDeviation` is a multiplier for deviation, default is 1;

* :math:`gradInfl` is a multiplier for gradient component of deviation, default is 1;

* :math:`allowErr` is a tolerance for the original cross-section. Means that cross-section getter retuns result for :math:`z=value±allowErr`.


Member functions
^^^^^^^^^^^^^^^^

Public:
-------

``GetCrossSectionOriginal(Eigen::MatrixXd & CSOmatTarget, double value, bool isCScomputed)`` writes a cross-section :math:`z=value±allowErr` matrix without deviation to the ``CSOmatTarget`` matrix. Parameter ``isCScomputed`` helps to avoid compiting the same matrix twice.

``GetCrossSectionExtended(Eigen::MatrixXd & CSEmatTarget, double value, int deviation, bool isCScomputed)`` writes a cross-section :math:`z=value±allowErr` matrix with the given deviation to the ``CSEmatTarget`` matrix. Parameter ``isCScomputed`` helps to avoid compiting the same matrix twice.

``GetCrossSectionExtendedAutoDev(Eigen::MatrixXd& CSEADmatTarget, double value)`` writes a cross-section :math:`z=value±allowErr` matrix with the deviation computed automaticly by ``ComputeCurrentDeviation()`` function to the ``CSEADmatTarget`` matrix.

``GetInterestingPoints(Eigen::MatrixXd & IPTarget)`` writes an array of interesting points to the ``IPTarget`` matrix.

``GetModifiedCrossSection(Eigen::MatrixXd & CSMTarget)`` is just getter for exteded cross-section (with auto deviation or not) without recomputation.

Protected:
----------

``ComputeGradient(double xStep, double yStep)`` computes gradients at every point for function values matrix with the given grid step:

.. math::
  \frac {f(x_i,y_i) - f(x_{i-1}, y_i)} {x_{i} - x_{i-1}},  \frac {f(x_i,y_i) - f(x_i, y_{i-1})} {y_{i} - y_{i-1}}

.. math:: 
  xStep = x_{i} - x_{i-1}

.. math::
  yStep = y_{i} - y_{i-1}

``addPoints(int deviation)`` finds points of extended cross-section with the given deviation.

.. uml:: 
   
   @startuml
	while(For each point in matrix)
		if (Cross-section contains current point) then (yes)
			if (Interesting points array contains this point) then (no)
				:Add point to interesting points array;
				:Set the corresponding extended cross-section matrix point 1;
			endif
			:makeCorridor(current_x, current_y, deviation);
			
		endif
	endwhile
        stop
   @enduml

``ComputeCrossSectionOriginal(double value)`` computes cross-section :math:`z = value ± allowErr`

``int ComputeCurrentDeviation()``

Algorithm:

* Compute non-zero elements in original cross-section;

* For each element in matrixes compute respectivetly: :math:`dx[i, j] * CrossSectionOriginal[i, j]` and :math:`dy[i, j] * CrossSectionOriginal[i, j]` (this will leave only contour's gradient points);

* Find the sqrt of sum of squares (to fing the length of gradient vector);

* Sum all this values;

* Divide it by the number of non-zero values to find the avarage value of contour's gradient;

* Multiply it with the gradient multiplier (can be set at constructor, default is 1); 

* Product with multiplier InitialDeviation (can be set at constructor, default is 1).

Returns the value of deviation.


``makeCorridor(int curr_x, int curr_y, int deviation)`` is auxiliary function for ``addPoints(int deviation)``. It fills the nearest points to the current points in extended cross-section matrix. As inputs there are coordinates of current point: :math:`curr\_x` and :math:`curr\_y` and deviation value that means that the square with side :math:`2 deviation + 1` will be considered. 

.. uml::

   @startuml
   while (For each the nearest point)
   if (Cross-section contains current point) then (yes)
   	if (Interesting points array contains this point) then (no)
   		:Add point to interesting points array;
   		:Set the corresponding extended cross-section matrix point 1;
   	endif
   endif
   endwhile
   stop
   @enduml
