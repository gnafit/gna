#ifndef PORABOLOID_H
#define PORABOLOID_H

#include <cmath>
#include <iostream>
#include <Eigen/Core>
#include "SpectrumCrossSection.hh"
#include <string>
using namespace std;
using namespace Eigen;

class Poraboloid
{
public:
	Poraboloid(Eigen::MatrixXd& mat, int initDeviation = 1, double allowErr = 0.0) 
		: PoraboloidMatrix(mat), InitialDeviation(initDeviation), AllowableError(allowErr) {
		PMrows = mat.rows();
		PMcols = mat.cols();
		ComputeGradient();
	}

	/**
	*
	* Returns cross-section z = value of PoraboloidMatrix: the plain contains contour
	* \param[in] value The value for compute cross-section plane z = value
	* \return Matrix contains values 0 or 1, where 1 means original contour point
	*
	*/
	MatrixXd GetCrossSectionOriginal(double value);

	/**
	*
	* Returns cross-section plane z = value of PoraboloidMatrix with the extended contour
	* \param[in] value The value for compute cross-section plane z = value
	* \param[in] deviation Deviation of the original contour
	* \return Matrix contains values 0 or 1, where 1 means extended contour point
	*
	*/
	MatrixXd GetCrossSectionExtended (double value, double deviation, bool isCScomuted = false);

	MatrixXd GetCrossSectionExtendedAutoDev (double value, string str);

protected:

	void ComputeGradient();
	void ComputeCrossSectionOriginal(double value); 
	int ComputeCurrentDeviation();

	MatrixXd PoraboloidMatrix;      //!< Full values matrix (2D and unknown size NxM)
	MatrixXd CrossSecOriginal; 	//!< Cross-section z=value, is not set at initial moment, can be recomputed
        MatrixXd dxPM, dyPM;            //!< Components of gradient for PoraboloidMatrix: sizes of [NxM-1] and [N-1xM]
        int InitialDeviation;           //!< Multiplier for deviation value, can be set at constructor, default is 1
	double AllowableError;		//!< In cross-section z = value finding there is z = value+-AllowableError is found in fact 
	int PMcols, PMrows;		//!< Size of PoraboloidMatrix, computed in constructor, can't be changed after

};

#endif /* PORABOLOID_H */
