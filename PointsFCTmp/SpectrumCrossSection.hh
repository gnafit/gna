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
	template <typename Derived>
	SpectrumCrossSection(Eigen::MatrixBase<Derived> const & mat) : CrossSection(mat), CorridorSize(0) {
		if (!checkInputOK()) std::cerr << "Incorrect input data: cross-section matrix must contain 0 and 1 only!";
		CrossSectionModified = Eigen::MatrixXd::Zero(CrossSection.rows(), CrossSection.cols());
	}

	void addPoints();

	/**
	* Setter for SpectrumCrossSection#CorridorSize
	*/
	inline void SetCorridor(int val) 		 { CorridorSize = val; }

	/**
	* Getter for SpectrumCrossSection#InterestingPoints matrix
	* \warning Should be computed at least once by addPoints() function before getting
	* \return SpectrumCrossSection#InterestingPoints matrix
	*/
	inline void GetInterestingPoints(Eigen::MatrixXd & IPTarget)   {  IPTarget = InterestingPoints; }
	
	/**
	* Getter for SpectrumCrossSection#CrossSectionModified matrix
	* \warning Should be computed at least once by addPoints() function before getting
	* \return SpectrumCrossSection#CrossSectionModified matrix
	*/
	inline void GetModifiedCrossSection(Eigen::MatrixXd & CSMTarget) { 
		CSMTarget = CrossSectionModified; 
	}

	
protected:
	bool checkInputOK();
	void ShowFoundPoints();
	void makeCorridor(int curr_x, int curr_y);

	Eigen::MatrixXd CrossSection;          			//!< Original, shouldn't be modified.
        Eigen::MatrixXd CrossSectionModified;  			//!< Output matrix, schould be modified. Zeros at the initial moment.
        int CorridorSize;               			//!< Deviation, zero at the initial moment.
        Eigen::Matrix2Xd InterestingPoints;  			//!< Found points

};

#endif /* SPECTRUMCROSSSECTION_H */
