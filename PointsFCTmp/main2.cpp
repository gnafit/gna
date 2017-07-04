#include <iostream>
#include <fstream>
#include <Eigen/Core>
#include "SpectrumCrossSection.hh"
#include "Poraboloid.hh"
#include <string>
/*#include "H5Cpp.h"
#include "hdf5.h"
using namespace H5;

const H5std_string	FILE_NAME("mytestfile.hdf5");
const H5std_string	DATASET_NAME("mydataset");
*/
const int 	DIM0 = 201;	               // dataset dimensions
const int 	DIM1 = 101;
//const int   RANK = 2;

using namespace Eigen;
using namespace std;

int main()
{
/*	double data[DIM0][DIM1];
        H5File file1( FILE_NAME, H5F_ACC_RDONLY );
        DataSet dataset1 = file1.openDataSet( DATASET_NAME );
	DataSpace filespace = dataset1.getSpace();
	int rank = filespace.getSimpleExtentNdims();
	hsize_t dims[2];    // dataset dimensions
	rank = filespace.getSimpleExtentDims( dims );
    
     * Define the memory space to read dataset.
     
        DataSpace mspace1(RANK, dims);
	dataset1.read( data, PredType::NATIVE_DOUBLE, mspace1, filespace );
	std::cout << data;
*/

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
	MatrixXd a(4, 4);
        a  << 5 , 5, 5, 0,
             0, 1, 0, 0,
             2, 0, 2, 0,
	     0, 0, 0, 0;
//      std::cout << a;
	//SpectrumCrossSection test(a);

//	test.SetCorridor(1);
//	test.addPoints();
        Poraboloid t(bbb, 1, 0.9);
	t.GetCrossSectionExtendedAutoDev(20, "out20_1_09.txt");

	//Poraboloid t2(bbb, 1, 0.9);
        t.GetCrossSectionExtendedAutoDev(80, "out80_1_09.txt");

	//Poraboloid t3(bbb, 1, 0.9);
        t.GetCrossSectionExtendedAutoDev(160, "out160_1_09.txt");

	t.GetCrossSectionExtended(160, 2);




//t.GetCrossSectionByValue(2, 1);

	//t.GetGradient();
       // t.CutCrossSection(3);
}

