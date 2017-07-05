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


MatrixXd Paraboloid::GetCrossSectionOriginal(double value, bool isCScomputed) {
/**
*
* Returns cross-section z = value of ParaboloidMatrix: the plain contains contour
* \param[in] value The value for compute cross-section plane z = value
* \param[in] isCScomputed It is false if CrossSecOriginal is not computed yet. It is necessary to avoid computing twice and ensure that it is not rubbish in this matrix
* \return Matrix contains values 0 or 1, where 1 means original contour point
*
*/
	if (! isCScomputed)  ComputeCrossSectionOriginal(value);
	return CrossSecOriginal;
}

void Paraboloid::ComputeCrossSectionOriginal(double value) {
std::cout << "I am computed!!!" << std::endl;
	CrossSecOriginal = ParaboloidMatrix.unaryExpr(Cutter<double>(value, AllowableError));
}

MatrixXd Paraboloid::GetCrossSectionExtended(double value, double deviation, bool isCScomputed) {
/**
*
* Returns cross-section plane z = value of ParaboloidMatrix with the extended contour
* \param[in] value The value for compute cross-section plane z = value
* \param[in] deviation Deviation of the original contour
* \param[in] isCScomputed It is false if CrossSecOriginal is not computed yet. It is necessary to avoid computing twice and ensure that it is not rubbish in this matrix
* \return Matrix contains values 0 or 1, where 1 means extended contour point
*
*/
	SpectrumCrossSection crossSec(GetCrossSectionOriginal(value, isCScomputed));
std::cout << " deviation = " << deviation << std::endl;
	crossSec.SetCorridor(deviation);
        crossSec.addPoints();
	return crossSec.GetModifiedCrossSection();
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

MatrixXd Paraboloid::GetCrossSectionExtendedAutoDev (double value, string str) {
/**
*
* Returns cross-section z = value of ParaboloidMatrix (the plain contains contour) using value only. The deviation is computed automaticly and depends on gradient at contour points.
* \param[in] value The value for compute cross-section plane z = value
* \return Matrix contains values 0 or 1, where 1 means extended contour point
*
*/
	ComputeCrossSectionOriginal(value);
	MatrixXd res = GetCrossSectionExtended(value, ComputeCurrentDeviation(), true);
    if( !str.empty() ) {
        std::ofstream file1;
        file1.open(str.c_str());
        file1 << res;
        file1.close();
    }
	return res;
}
