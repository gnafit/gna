Paraboloid
~~~~~~~~~~

Description
^^^^^^^^^^^

Contais methods for working with cross-sections of the predetermined function. Function is represented as a matrix with values computed with the given grid.

The following operations can be done:

* Get cross-section :math:`z=value` with the given function value;

* Get extended cross-section with the given function value and deviation from the original cross-section contour. Extended cross-section contains :math:`|z-value| <= deviation` points;

* Get extended cross-section with the given function value where deviation is computed automaticly and depends on the gradient vector length at cross-section points.

Constructors
^^^^^^^^^^^^


``Paraboloid(Eigen::MatrixXd & mat, double xStep, double yStep, int initDeviation, double gradInfl, double allowErr)``

``Paraboloid(int rows, int columns, double* mat, double xStep, double yStep, int initDeviation, double gradInfl, double allowErr)``

where 

* :math:`mat` is matrix contains float values;

* :math:`rows` and :math:`columns` are sizes of matrix :math:`mat`;

* :math:`xStep` and :math:`yStep` are grid steps;

* :math:`initDeviation` is a multiplier for deviation, default is 1;

* :math:`gradInfl` is a multiplier for gradient component of deviation, default is 1;

* :math:`allowErr` is a tolerance for the original cross-section. Means that cross-section getter retuns result for :math:`z=value+-allowErr`.


Member functions
^^^^^^^^^^^^^^^^

``ComputeGradient(double xStep, double yStep)`` computes gradients at every point for paraboloid matrix with the given grid step:

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
			while (For each nearest point)
				if (Cross-section contains current point) then (yes)
					if (Interesting points array contains this point) then (no)
						:Add point to interesting points array;
						:Set the corresponding extended cross-section matrix point 1;
					endif
				endif
			endwhile
		endif
	endwhile
   @enduml

``ComputeCrossSectionOriginal(double value)`` computes cross-section :math:`z=value+-allowErr`

``int ComputeCurrentDeviation()``

``void makeCorridor(int curr_x, int curr_y, int deviation)``



