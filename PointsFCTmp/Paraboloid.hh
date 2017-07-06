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

	template <typename Derived>
	Paraboloid(Eigen::MatrixBase<Derived> const & mat, int initDeviation = 1, double allowErr = 0.0)
		: ParaboloidMatrix(mat), InitialDeviation(initDeviation), AllowableError(allowErr) {
		PMrows = mat.rows();
		PMcols = mat.cols();
		ComputeGradient();
	}

	Paraboloid(int rows, int columns, double* mat, int initDeviation = 1, double allowErr = 0.0)
		: ParaboloidMatrix(Eigen::Map<Eigen::MatrixXd>(mat, rows, columns)),
		  InitialDeviation(initDeviation), AllowableError(allowErr) {
	        PMrows = rows;
       		PMcols = columns;
        	ComputeGradient();
	}


	template <typename Derived>
	void GetCrossSectionOriginal(Eigen::MatrixBase<Derived> const & CSOmatTarget, double value, bool isCScomputed = false) {
	/**
	*
	* Returns cross-section z = value of ParaboloidMatrix: the plain contains contour
	* \param[in] CSOmatTarget The matrix where result will be written
	* \param[in] value The value for compute cross-section plane z = value
	* \param[in] isCScomputed It is false if CrossSecOriginal is not computed yet. It is necessary to avoid computing twice and ensure that it is not rubbish in this matrix
	*
	*/
       		if (! isCScomputed)  ComputeCrossSectionOriginal(value);

  		Eigen::MatrixBase<Derived>& C = const_cast< Eigen::MatrixBase<Derived>& >(CSOmatTarget);
		C = CrossSecOriginal;	
	}

	template <typename Derived>
	void GetCrossSectionExtended (Eigen::MatrixBase<Derived> const & CSEmatTarget, 
					double value, double deviation, bool isCScomputed = false) {
	/**
	*
	* Returns cross-section plane z = value of ParaboloidMatrix with the extended contour
	* \param[in] CSEmatTarget The matrix where result will be written
	* \param[in] value The value for compute cross-section plane z = value
	* \param[in] deviation Deviation of the original contour
	* \param[in] isCScomputed It is false if CrossSecOriginal is not computed yet. It is necessary to avoid computing twice and ensure that it is not rubbish in this matrix
	*
	*/

        	Eigen::MatrixXd tmpMat;
        	GetCrossSectionOriginal(tmpMat, value, isCScomputed);
        	SpectrumCrossSection crossSec(tmpMat);

		std::cout << " deviation = " << deviation << std::endl;
        	crossSec.SetCorridor(deviation);
        	crossSec.addPoints();
        	crossSec.GetModifiedCrossSection(CSEmatTarget);
	}

	template <typename Derived>
	void GetCrossSectionExtendedAutoDev (Eigen::MatrixBase<Derived> const & CSEADmatTarget, double value) {
	/**
	*
	* Returns cross-section z = value of ParaboloidMatrix (the plain contains contour) using value only. The deviation is computed automaticly and depends on gradient at contour points.
	* \param[in] CSEADmatTarget The matrix where result will be written
	* \param[in] value The value for compute cross-section plane z = value
	*
	*/
		ComputeCrossSectionOriginal(value);
        	GetCrossSectionExtended(CSEADmatTarget, value, ComputeCurrentDeviation(), true);
	}

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
