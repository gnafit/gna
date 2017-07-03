#ifndef SPECTRUMCROSSSECTION_H
#define SPECTRUMCROSSSECTION_H

#include <cmath>
#include <iostream>
#include <Eigen/Core>
#include <list>

using namespace Eigen;


class SpectrumCrossSection
{
public:
	/**
	* Constructor: matrix NxM that contain 0 and 1 only is expected as input 
	* \todo: check if there only 0 and 1
	*/
	SpectrumCrossSection(MatrixXd mat) : CrossSection(mat), CorridorSize(0) {
		CrossSectionModified = MatrixXd::Zero(CrossSection.rows(), CrossSection.cols());
	}
	
	void addPoints();

	inline void SetCorridor(int val) { CorridorSize = val; }
	inline std::list<Vector2d> GetInterestingPoints() { return InterestingPoints; }
	inline MatrixXd GetModifiedCrossSection() { return CrossSectionModified; }


protected:
	void makeCorridor(int curr_x, int curr_y);
	MatrixXd CrossSection;          //!< Original, shouldn't be modified.
        MatrixXd CrossSectionModified;  //!< Output matrix, schould be modified. Zeros at the initial moment.
        int CorridorSize;               //!< Deviation, zero at the initial moment.
        std::list<Vector2d> InterestingPoints;  //!< List of found points

};

#endif /* SPECTRUMCROSSSECTION_H */
