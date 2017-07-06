#include "Paraboloid.hh"
#include <Eigen/Core>
#include <fstream>
#include <cmath>
#include <string>
using namespace std;
using namespace Eigen;

/**
* The only one goal of creating this structure is to optimize matrix openation
*/
template<typename T>
struct Cutter {
  Cutter(const T& val, const T& err) : v(val), e(err) {}
  const T operator()(const T& x) const { return std::abs(x - v) <= e ? 1.0 : 0.0; }
  T v, e;
};

void Paraboloid::ComputeCrossSectionOriginal(double value) {
	CrossSectionModified = Eigen::MatrixXd::Zero(PMrows, PMcols);
	std::cout << "I am computed!!!" << std::endl;
	CrossSecOriginal = ParaboloidMatrix.unaryExpr(Cutter<double>(value, AllowableError));
}


void Paraboloid::ComputeGradient() {
/**
*
* Computes dxPM and dyPM (components of gradient)
* Gradient matrixes are the folowing
*
*	- dx - size [NxM-1]
*	- dy - size [N-1xM]
*
* as gradient is computed with the neighbour elements in matrix
*
*/
	dxPM = ParaboloidMatrix.rightCols(PMcols - 1) - ParaboloidMatrix.leftCols(PMcols - 1);
	dyPM = ParaboloidMatrix.bottomRows(PMrows - 1) - ParaboloidMatrix.topRows(PMrows - 1);
}

int Paraboloid::ComputeCurrentDeviation() {
/**
*
* Algorithm:
*	- Compute non-zero elements in original cross-section
*	- For each element in matrixes compute respectivetly:
*	  dx[i, j]*CrossSectionOriginal[i, j] and dy[i, j]*CrossSectionOriginal[i, j]
*	  (this will leave only contour's gradient points)
*	- Find the sqrt of sum of squares (to fing the length of gradient vector)
*	- Sum all this values and divide by the number of non-zero values to find the avarage value of contour's gradient
*	- Product with multiplier InitialDeviation
*
* \return Deviation from the original coutour - the number of points that will be included in extended contour.
*
*/

	int rowsnum = PMrows - 1, colsnum = PMcols - 1;
	int numOfNonZero = (CrossSecOriginal.array() != 0).count();
	std::cout << "numOfNonZero = "  << numOfNonZero << std::endl;
	double  tmp =  ((dxPM.block(0, 0, rowsnum, colsnum).array() *
				CrossSecOriginal.block(0, 0, rowsnum, colsnum).array()).square() +
			(dyPM.block(0, 0, rowsnum, colsnum).array() *
                		CrossSecOriginal.block(0, 0, rowsnum, colsnum).array()).square())
				.sqrt().sum() / numOfNonZero;
	return std::ceil(tmp) * InitialDeviation;
}

void Paraboloid::GetCrossSectionOriginal(Eigen::MatrixXd& CSOmatTarget, double value, bool isCScomputed ) {
        /**
        *
        * Returns cross-section z = value of ParaboloidMatrix: the plain contains contour
        * \param[in] CSOmatTarget The matrix where result will be written
        * \param[in] value The value for compute cross-section plane z = value
        * \param[in] isCScomputed It is false if CrossSecOriginal is not computed yet. It is necessary to avoid computing twice and ensure that it is not rubbish in this matrix
        *
        */
                if (! isCScomputed)  ComputeCrossSectionOriginal(value);

              //  Eigen::MatrixBase<Derived>& C = const_cast< Eigen::MatrixBase<Derived>& >(CSOmatTarget);
                CSOmatTarget = CrossSecOriginal;
        }


void Paraboloid::GetCrossSectionExtended(MatrixXd & CSEmatTarget,
                                        double value, double deviation, bool isCScomputed) {
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
               // SpectrumCrossSection crossSec(tmpMat);

                std::cout << " deviation = " << deviation << std::endl;
               // SetCorridor(deviation);
                addPoints(deviation);
                GetModifiedCrossSection(CSEmatTarget);
        }


void Paraboloid::GetCrossSectionExtendedAutoDev(Eigen::MatrixXd& CSEADmatTarget, double value) {
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

void Paraboloid::makeCorridor(int curr_x, int curr_y, int deviation) {
/**
* Adds neighbour points to the InterestingPoints matrix for point (curr_x, curr_y)
* \param[in] curr_x x-coordinate of the considered point
* \param[in] curr_y y-coordinate of the considered point
*/
	for (int i = curr_x - deviation; i < curr_x + deviation + 1; i++) {
		if (i < 0 || i >=  PMrows) continue;
		for (int j = curr_y - deviation; j < curr_y + deviation + 1; j++) {
			if (j <  0 || j >= PMcols) continue;
			if (CrossSecOriginal(i, j) == 1.0) continue;
			InterestingPoints.conservativeResize(2, InterestingPoints.cols() + 1);
			InterestingPoints(0, InterestingPoints.cols() - 1) = i;
			InterestingPoints(1, InterestingPoints.cols() - 1) = j;
			CrossSectionModified(i, j) = 1.0;
		}
	}

}

void Paraboloid::addPoints(int deviation) {
/**
* Fills Paraboloid#InterestingPoints matrix by extended contour points. 
*/
	InterestingPoints.resize(2, 0);
	for(int i = 0; i < PMrows; i++) {
		for (int j = 0; j < PMcols; j++) {
			if (CrossSecOriginal(i, j) == 1.0) {
				InterestingPoints.conservativeResize(2, InterestingPoints.cols() + 1);
	                        InterestingPoints(0, InterestingPoints.cols() - 1) = i;
        	                InterestingPoints(1, InterestingPoints.cols() - 1) = j;
				CrossSectionModified(i, j) = 1.0;
				makeCorridor(i, j, deviation);
			}
		}
	}
}

