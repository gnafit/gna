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

	/**
	*
	* Returns cross-section z = value of ParaboloidMatrix: the plain contains contour
	* \param[in] value The value for compute cross-section plane z = value
	* \return Matrix contains values 0 or 1, where 1 means original contour point
	*
	*/
    Eigen::MatrixXd GetCrossSectionOriginal(double value);

	/**
	*
	* Returns cross-section plane z = value of ParaboloidMatrix with the extended contour
	* \param[in] value The value for compute cross-section plane z = value
	* \param[in] deviation Deviation of the original contour
	* \return Matrix contains values 0 or 1, where 1 means extended contour point
	*
	*/
	Eigen::MatrixXd GetCrossSectionExtended (double value, double deviation, bool isCScomuted = false);

	Eigen::MatrixXd GetCrossSectionExtendedAutoDev (double value, std::string str);

protected:

	void ComputeGradient();
	void ComputeCrossSectionOriginal(double value);
	int ComputeCurrentDeviation();

	Eigen::MatrixXd ParaboloidMatrix;      //!< Full values matrix (2D and unknown size NxM)
	Eigen::MatrixXd CrossSecOriginal; 	//!< Cross-section z=value, is not set at initial moment, can be recomputed
        Eigen::MatrixXd dxPM, dyPM;            //!< Components of gradient for ParaboloidMatrix: sizes of [NxM-1] and [N-1xM]
        int InitialDeviation;           //!< Multiplier for deviation value, can be set at constructor, default is 1
	double AllowableError;		//!< In cross-section z = value finding there is z = value+-AllowableError is found in fact
	int PMcols, PMrows;		//!< Size of ParaboloidMatrix, computed in constructor, can't be changed after

};

#endif /* PARABOLOID_H */
