#ifndef SPECTRUMCROSSSECTION_H
#define SPECTRUMCROSSSECTION_H

#include <cmath>
#include <iostream>
#include <Eigen/Core>

class SpectrumCrossSection
{
public:
	/**
	* Constructor: matrix NxM that contain 0 and 1 only is expected as input
	*/
	SpectrumCrossSection(Eigen::MatrixXd mat) : CrossSection(mat), CorridorSize(0) {
		if (!checkInputOK()) std::cerr << "Incorrect input data: cross-section matrix must contain 0 and 1 only!";
		CrossSectionModified = Eigen::MatrixXd::Zero(CrossSection.rows(), CrossSection.cols());
	}

	void addPoints();

	inline void SetCorridor(int val) { CorridorSize = val; }
	inline Eigen::Matrix2Xd GetInterestingPoints() { return InterestingPoints; }
	inline Eigen::MatrixXd GetModifiedCrossSection() { return CrossSectionModified; }


protected:
	bool checkInputOK();
	void ShowFoundPoints();
	void makeCorridor(int curr_x, int curr_y);
	Eigen::MatrixXd CrossSection;          			//!< Original, shouldn't be modified.
        Eigen::MatrixXd CrossSectionModified;  			//!< Output matrix, schould be modified. Zeros at the initial moment.
        int CorridorSize;               			//!< Deviation, zero at the initial moment.
        Eigen::Matrix2Xd InterestingPoints;  				//!< List of found points

};

#endif /* SPECTRUMCROSSSECTION_H */
