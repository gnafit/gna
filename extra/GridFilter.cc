#include "GridFilter.hh"
#include <Eigen/Core>
#include <fstream>
#include <cmath>
#include <string>
using namespace std;
using namespace Eigen;

//#define DEBUG_GRIDFILTER

/**
* The only one goal of creating this structure is to optimize matrix openation
*/
template<typename T>
struct Cutter {
  Cutter(const T& val, const T& err) : v(val), e(err) {}
  const T operator()(const T& x) const { return std::abs(x - v) <= e ? 1.0 : 0.0; }
  T v, e;
};

void GridFilter::ComputeCrossSectionOriginal(double value) {
	m_CrossSectionModified = Eigen::MatrixXd::Zero(m_PMrows, m_PMcols);
    #ifdef DEBUG_GRIDFILTER
	std::cout << "I am computed!!!" << std::endl;
    #endif
	m_CrossSecOriginal = m_ParaboloidMatrix.unaryExpr(Cutter<double>(value, m_AllowableError));
}

void GridFilter::ComputeGradient(double xStep, double yStep) {
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
	m_dxPM = (m_ParaboloidMatrix.rightCols(m_PMcols - 1) - m_ParaboloidMatrix.leftCols(m_PMcols - 1)) / xStep;
	m_dyPM = (m_ParaboloidMatrix.bottomRows(m_PMrows - 1) - m_ParaboloidMatrix.topRows(m_PMrows - 1)) / yStep;
}

int GridFilter::ComputeCurrentDeviation() {
/**
*
* Algorithm:
*	- Compute non-zero elements in original cross-section
*	- For each element in matrixes compute respectivetly:
*	  dx[i, j]*CrossSectionOriginal[i, j] and dy[i, j]*CrossSectionOriginal[i, j]
*	  (this will leave only contour's gradient points)
*	- Find the sqrt of sum of squares (to fing the length of gradient vector)
*	- Sum all this values and divide by the number of non-zero values to find the avarage value of contour's gradient
*	- Product with multiplier GradientInfluence, ceil and product with InitialDeviation
*
* \return Deviation from the original coutour - the number of points that will be included in extended contour.
*
*/

	int rowsnum = m_PMrows - 1, colsnum = m_PMcols - 1;
	int numOfNonZero = (m_CrossSecOriginal.array() != 0).count();
	if (numOfNonZero == 0) {
		std::cerr << "No contour found on this level. If you are sure that it shold be here, try the following:" << std::endl
			  << "- make grid step smaller " << std::endl
			  << "- make tolerance higher" << std::endl;
		return 0;
	}
    #ifdef DEBUG_GRIDFILTER
	std::cout << "numOfNonZero = "  << numOfNonZero << std::endl;
    #endif
	double  tmp =  ((m_dxPM.block(0, 0, rowsnum, colsnum).array() *
				m_CrossSecOriginal.block(0, 0, rowsnum, colsnum).array()).square() +
			(m_dyPM.block(0, 0, rowsnum, colsnum).array() *
                		m_CrossSecOriginal.block(0, 0, rowsnum, colsnum).array()).square())
				.sqrt().sum() * m_GradientInfluence / numOfNonZero;
        #ifdef DEBUG_GRIDFILTER
        std::cout << "grad_len = "  << tmp << std::endl;
        #endif
	return std::ceil(tmp) * m_InitialDeviation;
}

void GridFilter::GetCrossSectionOriginal(Eigen::MatrixXd& CSOmatTarget, double value, bool isCScomputed ) {
/**
*
* Returns cross-section z = value of ParaboloidMatrix: the plain contains contour
* \param[in] CSOmatTarget The matrix where result will be written
* \param[in] value The value for compute cross-section plane z = value
* \param[in] isCScomputed It is false if CrossSecOriginal is not computed yet. It is necessary to avoid computing twice and ensure that it is not rubbish in this matrix
*
*/
        if (! isCScomputed)  ComputeCrossSectionOriginal(value);
        CSOmatTarget = m_CrossSecOriginal;
}


void GridFilter::GetCrossSectionExtended(MatrixXd & CSEmatTarget,
                                        double value, int deviation, bool isCScomputed) {
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
        #ifdef DEBUG_GRIDFILTER
        std::cout << " deviation = " << deviation << std::endl;
        #endif
        if (deviation != 0) addPoints(deviation);
	else {
		CSEmatTarget = MatrixXd::Zero(m_PMrows, m_PMcols);
	}
	GetModifiedCrossSection(CSEmatTarget);
}

void GridFilter::GetCrossSectionExtendedAutoDev(Eigen::MatrixXd& CSEADmatTarget, double value) {
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

void GridFilter::makeCorridor(int curr_x, int curr_y, int deviation) {
/**
* Adds neighbour points to the InterestingPoints matrix for point (curr_x, curr_y)
* \param[in] curr_x x-coordinate of the considered point
* \param[in] curr_y y-coordinate of the considered point
* \param[in] deviation Deviation from the original contour
*/
	for (int i = curr_x - deviation; i < curr_x + deviation + 1; i++) {
		if (i < 0 || i >=  m_PMrows) continue;
		for (int j = curr_y - deviation; j < curr_y + deviation + 1; j++) {
			if (j <  0 || j >= m_PMcols) continue;
			if (m_CrossSecOriginal(i, j) == 1.0) continue;
			m_InterestingPoints.conservativeResize(2, m_InterestingPoints.cols() + 1);
			m_InterestingPoints(0, m_InterestingPoints.cols() - 1) = i;
			m_InterestingPoints(1, m_InterestingPoints.cols() - 1) = j;
			m_CrossSectionModified(i, j) = 1.0;
		}
	}

}

void GridFilter::addPoints(int deviation) {
/**
* Fills GridFilter#InterestingPoints matrix by extended contour points.
*/
	m_InterestingPoints.resize(2, 0);
	for(int i = 0; i < m_PMrows; i++) {
		for (int j = 0; j < m_PMcols; j++) {
			if (m_CrossSecOriginal(i, j) == 1.0) {
				m_InterestingPoints.conservativeResize(2, m_InterestingPoints.cols() + 1);
	                        m_InterestingPoints(0, m_InterestingPoints.cols() - 1) = i;
        	                m_InterestingPoints(1, m_InterestingPoints.cols() - 1) = j;
				m_CrossSectionModified(i, j) = 1.0;
				makeCorridor(i, j, deviation);
			}
		}
	}
}

