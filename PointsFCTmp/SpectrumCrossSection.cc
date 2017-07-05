#include "SpectrumCrossSection.hh"
#include <fstream>

using namespace Eigen;
using namespace std;

void SpectrumCrossSection::makeCorridor(int curr_x, int curr_y) {
/**
* Adds neighbour points to the InterestingPoints matrix for point (curr_x, curr_y)
* \param[in] curr_x x-coordinate of the considered point
* \param[in] curr_y y-coordinate of the considered point
*/
	for (int i = curr_x - CorridorSize; i < curr_x + CorridorSize + 1; i++) {
		if (i < 0 || i >=  CrossSection.rows()) continue;
		for (int j = curr_y - CorridorSize; j < curr_y + CorridorSize + 1; j++) {
			if (j <  0 || j >= CrossSection.cols()) continue;
			if (CrossSection(i, j) == 1.0) continue;
			InterestingPoints.conservativeResize(2, InterestingPoints.cols() + 1);
			InterestingPoints(0, InterestingPoints.cols() - 1) = i;
			InterestingPoints(1, InterestingPoints.cols() - 1) = j;
			CrossSectionModified(i, j) = 1.0;
		}
	}

}

void SpectrumCrossSection::addPoints() {
/**
* Fills InterestingPoints matrix by extended contour points. 
*/
	InterestingPoints.resize(2, 0);
	for(int i = 0; i < CrossSection.rows(); i++) {
		for (int j = 0; j < CrossSection.cols(); j++) {
			if (CrossSection(i, j) == 1.0) {
				InterestingPoints.conservativeResize(2, InterestingPoints.cols() + 1);
	                        InterestingPoints(0, InterestingPoints.cols() - 1) = i;
        	                InterestingPoints(1, InterestingPoints.cols() - 1) = j;
				CrossSectionModified(i, j) = 1.0;
				makeCorridor(i, j);
			}
		}
	}
	ShowFoundPoints();
}

void SpectrumCrossSection::ShowFoundPoints() {
	ofstream file;
	file.open("points.txt");
	std::cout << "cols of InterestingPoints = " << InterestingPoints.cols() << std::endl;
	file << InterestingPoints << std::endl;
	file.close();
}

bool SpectrumCrossSection::checkInputOK() {
	return (CrossSection.array() != 1 && CrossSection.array() != 0).count() == 0 ? true : false;
}

