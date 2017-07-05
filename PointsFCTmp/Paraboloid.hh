#ifndef PARABOLOID_H
#define PARABOLOID_H

#include <cmath>
#include <iostream>
#include <Eigen/Core>
#include "SpectrumCrossSection.hh"
#include <string>

class Paraboloid
{
public:
	Paraboloid(Eigen::MatrixXd& mat, int initDeviation = 1, double allowErr = 0.0)
		: ParaboloidMatrix(mat), InitialDeviation(initDeviation), AllowableError(allowErr) {
		PMrows = mat.rows();
		PMcols = mat.cols();
		ComputeGradient();
	}

	Paraboloid(int rows, int columns, double* mat, int initDeviation = 1, double allowErr = 0.0)
		: ParaboloidMatrix(Eigen::Map<Eigen::MatrixXd>(mat, rows, columns)), InitialDeviation(initDeviation), AllowableError(allowErr) {
        PMrows = rows;
        PMcols = columns;
        ComputeGradient();
	}


	// Returns cross-section z = value of ParaboloidMatrix
	Eigen::MatrixXd GetCrossSectionOriginal(double value, bool isCScomputed = false);

	// Returns cross-section plane z = value of ParaboloidMatrix with the extended contour
	Eigen::MatrixXd GetCrossSectionExtended (double value, double deviation, bool isCScomputed = false);

	Eigen::MatrixXd GetCrossSectionExtendedAutoDev (double value, std::string str="");

protected:

	void ComputeGradient();
	void ComputeCrossSectionOriginal(double value);
	int ComputeCurrentDeviation();

	Eigen::MatrixXd ParaboloidMatrix;	//!< Full values matrix (2D and unknown size NxM)
	Eigen::MatrixXd CrossSecOriginal; 	//!< Cross-section z=value, is not set at initial moment, can be recomputed
        Eigen::MatrixXd dxPM, 			//!< x-component of gradient for ParaboloidMatrix size of [NxM-1]
			dyPM;           	//!< y-omponents of gradient for ParaboloidMatrix size od [N-1xM]
        int InitialDeviation;           	//!< Multiplier for deviation value, can be set at constructor, default is 1
	double AllowableError;			//!< In cross-section z = value finding there is z = value+-AllowableError is found in fact
	int PMcols, 				//!< The number of columns of ParaboloidMatrix, computed in constructor, can't be changed after
	    PMrows;				//!< The number of rows of ParaboloidMatrix, computed in constructor, can't be changed after

};

#endif /* PARABOLOID_H */
