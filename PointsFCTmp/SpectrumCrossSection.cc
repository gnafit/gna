#include "SpectrumCrossSection.hh"
#include <iterator>

using namespace Eigen;
using namespace std;

void SpectrumCrossSection::makeCorridor(int curr_x, int curr_y) {
	for (int i = curr_x - CorridorSize; i < curr_x + CorridorSize + 1; i++) {
		if (i < 0 || i >=  CrossSection.rows()) continue;
		for (int j = curr_y - CorridorSize; j < curr_y + CorridorSize + 1; j++) {
			if (j <  0 || j >= CrossSection.cols()) continue;
			if (CrossSection(i, j) == 1.0) continue;
			InterestingPoints.push_back(Vector2d(i, j));
			CrossSectionModified(i, j) = 1.0;
			//std::cout << " VECT " << std::endl << Vector2d(i, j); 
		}
	}

}

void SpectrumCrossSection::addPoints() {
	//std::cout << std::endl << "CrossSection " << std::endl << CrossSection << std::endl;

	for(int i = 0; i < CrossSection.rows(); i++) {
		for (int j = 0; j < CrossSection.cols(); j++) {
			if (CrossSection(i, j) == 1.0) {
				InterestingPoints.push_back(Vector2d(i, j));
				CrossSectionModified(i, j) = 1.0;
				makeCorridor(i, j);
			}
		}
	}
	//std::cout << std::endl << "CrossSectionM " << std::endl << CrossSectionModified << std::endl;
}
