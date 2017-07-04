#ifndef SPECTRUMCROSSSECTION_H
#define SPECTRUMCROSSSECTION_H

#include <cmath>
#include <iostream>
#include <Eigen/Core>

using namespace Eigen;

class SpectrumCrossSection
{
public:
	/**
	* Constructor: matrix NxM that contain 0 and 1 only is expected as input 
	*/
	SpectrumCrossSection(MatrixXd mat) : CrossSection(mat), CorridorSize(0) {
		if (!checkInputOK()) std::cerr << "Incorrect input data: cross-section matrix must contain 0 and 1 only!";
		CrossSectionModified = MatrixXd::Zero(CrossSection.rows(), CrossSection.cols());
	}
	
	void addPoints();

	inline void SetCorridor(int val) { CorridorSize = val; }
	inline Matrix2Xd GetInterestingPoints() { return InterestingPoints; }
	inline MatrixXd GetModifiedCrossSection() { return CrossSectionModified; }


protected:
	bool checkInputOK();
	void ShowFoundPoints();
	void makeCorridor(int curr_x, int curr_y);
	MatrixXd CrossSection;          			//!< Original, shouldn't be modified.
        MatrixXd CrossSectionModified;  			//!< Output matrix, schould be modified. Zeros at the initial moment.
        int CorridorSize;               			//!< Deviation, zero at the initial moment.
        Matrix2Xd InterestingPoints;  				//!< List of found points

};

#endif /* SPECTRUMCROSSSECTION_H */
