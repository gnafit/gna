#include <iostream>
#include <fstream>
#include <Eigen/Core>
#include "SpectrumCrossSection.hh"
#include "Poraboloid.hh"
#include <string>
const int 	DIM0 = 201;	               // dataset dimensions
const int 	DIM1 = 101;

using namespace Eigen;
using namespace std;

int main()
{

double x;
	MatrixXd bbb(DIM0, DIM1);
	std::ifstream myfile;
	myfile.open ("workfile");
for (int i = 0; i < DIM0; i++)
{
	for(int j = 0; j < DIM1; j++)
	{
		myfile >>  bbb(i, j);
	}
}
  	myfile.close();
//	std::cout << bbb << std::endl;


        Poraboloid t(bbb, 1, 0.9);
	t.GetCrossSectionExtendedAutoDev(20, "out20_1_09.txt");

	//Poraboloid t2(bbb, 1, 0.9);
        t.GetCrossSectionExtendedAutoDev(80, "out80_1_09.txt");

	//Poraboloid t3(bbb, 1, 0.9);
        t.GetCrossSectionExtendedAutoDev(160, "out160_1_09.txt");

	//t.GetCrossSectionByValue(2, 1);

	//t.GetGradient();
       // t.CutCrossSection(3);
}

