#include "Poraboloid.hh"
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


MatrixXd Poraboloid::GetCrossSectionOriginal(double value) {
	MatrixXd crossSec = PoraboloidMatrix;
	crossSec = PoraboloidMatrix.unaryExpr(Cutter<double>(value, AllowableError));
	return crossSec;
}

MatrixXd Poraboloid::GetCrossSectionExtended(double value, double deviation) {
	SpectrumCrossSection crossSec(GetCrossSectionOriginal(value));
std::cout << " deviation = " << deviation << std::endl;
	crossSec.SetCorridor(deviation);
        crossSec.addPoints();
	return crossSec.GetModifiedCrossSection();
}


void Poraboloid::ComputeGradient() {
/*
*
* Gradient matrixes are the folowing:
* dx - size [NxN-1], dy - size [N-1xN]
* as gradient is computed with the neighbour elements in matrix
*
*/

// TODO: delete file output

	std::ofstream file1, file2;
	file1.open("dxPM.txt");
	file2.open("dyPM.txt");
	dxPM = PoraboloidMatrix.rightCols(PoraboloidMatrix.cols()-1) - PoraboloidMatrix.leftCols(PoraboloidMatrix.cols()-1);  
	dyPM = PoraboloidMatrix.bottomRows(PoraboloidMatrix.rows()-1) - PoraboloidMatrix.topRows(PoraboloidMatrix.rows()-1);
	file1 << std::endl <<  "gradX = " << std::endl << dxPM << std::endl; 
	file2 << std::endl <<  "gradY = " << std::endl << dyPM << std::endl;
	file1.close(); file2.close();
}

int Poraboloid::ComputeCurrentDeviation(MatrixXd originalCrossSec) {
/**
*
*	- Compute non-zero elements in original cross-section
*	- For each element in matrixes compute respectivetly:
*	  dx[i, j]*CrossSectionOriginal[i, j] and dy[i, j]*CrossSectionOriginal[i, j]
*	  (this will leave only contour's gradient points)
*	- Find the sqrt of sum of squares (to fing the length of gradient vector)
*	- Sum all this values and divide by the number of non-zero values to find the avarage value of contour's gradient
*	- Product with multiplier InitialDeviation
*
*/	
	int numOfNonZero = (originalCrossSec.array() != 0).count();
std::cout << "numOfNonZero = "  << numOfNonZero << std::endl;
	double  tmp =  ((dxPM.block(0, 0, dxPM.rows() - 1, dxPM.cols()).array() * 
				originalCrossSec.block(0, 0, originalCrossSec.rows() - 1, originalCrossSec.cols() - 1).array()).square() + 
			(dyPM.block(0, 0, dyPM.rows(), dyPM.cols() - 1).array() * 
                		originalCrossSec.block(0, 0, originalCrossSec.rows() - 1, originalCrossSec.cols() - 1).array()).square())
				.sqrt().sum() / numOfNonZero;
	return std::ceil(tmp) * InitialDeviation;
}

MatrixXd Poraboloid::GetCrossSectionExtendedAutoDev (double value, string str) {
	MatrixXd res = GetCrossSectionExtended(value, ComputeCurrentDeviation(GetCrossSectionOriginal(value)));
	std::ofstream file1;
        file1.open(str.c_str());
	file1 << res;
	file1.close();
	/*TODO: Twice GetCrossSectionOriginal!!! Not optimal!!!*/
	return res;
}
