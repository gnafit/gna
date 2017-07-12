#ifndef GRIDFILTER_H
#define GRIDFILTER_H

#include <cmath>
#include <iostream>
#include <Eigen/Core>
#include <string>

class GridFilter
{
public:

	
	GridFilter(const Eigen::MatrixXd & mat, double xStep, double yStep, 
		int initDeviation = 1, double gradInfl = 1.0, double allowErr = 0.0)
		: m_ParaboloidMatrix(mat), m_InitialDeviation(initDeviation), m_GradientInfluence(gradInfl), m_AllowableError(allowErr) {
		m_PMrows = mat.rows();
		m_PMcols = mat.cols();
		ComputeGradient(xStep, yStep);
		m_CrossSectionModified = Eigen::MatrixXd::Zero(m_PMrows, m_PMcols);
	}

	GridFilter(int rows, int columns, double* mat, double xStep, double yStep, 
		int initDeviation = 1, double gradInfl = 1.0, double allowErr = 0.0)
		: m_ParaboloidMatrix(Eigen::Map<Eigen::MatrixXd>(mat, rows, columns)),
		  m_InitialDeviation(initDeviation), m_GradientInfluence(gradInfl), m_AllowableError(allowErr) {
	        m_PMrows = rows;
       		m_PMcols = columns;
        	ComputeGradient(xStep, yStep);
		m_CrossSectionModified = Eigen::MatrixXd::Zero(m_PMrows, m_PMcols);
	}

	void GetCrossSectionOriginal(Eigen::MatrixXd & CSOmatTarget, double value, bool isCScomputed = false);

	void GetCrossSectionExtended(Eigen::MatrixXd & CSEmatTarget, 
					double value, int deviation, bool isCScomputed = false);

	void GetCrossSectionExtendedAutoDev(Eigen::MatrixXd& CSEADmatTarget, double value);

	/**
	* Getter for GridFilter#InterestingPoints matrix
	* \warning Should be computed at least once by addPoints() function before getting
	* \return SpectrumCrossSection#InterestingPoints matrix
	*/
	inline void GetInterestingPoints(Eigen::MatrixXd & IPTarget)   {  IPTarget = m_InterestingPoints; }
	
	/**
	* Getter for GridFilter#CrossSectionModified matrix
	* \warning Should be computed at least once by addPoints() function before getting
	* \return GridFilter#CrossSectionModified matrix
	*/
	inline void GetModifiedCrossSection(Eigen::MatrixXd & CSMTarget) { CSMTarget = m_CrossSectionModified; }

protected:

	void ComputeGradient(double xStep, double yStep);
	void addPoints(int deviation);
	void ComputeCrossSectionOriginal(double value);
	int ComputeCurrentDeviation();
	void makeCorridor(int curr_x, int curr_y, int deviation);

	Eigen::MatrixXd m_ParaboloidMatrix;	//!< Full values matrix (2D and unknown size NxM)
	Eigen::MatrixXd m_CrossSecOriginal; 	//!< Cross-section z=value, is not set at initial moment, can be recomputed
	Eigen::MatrixXd m_CrossSectionModified;
        Eigen::MatrixXd m_dxPM, 		//!< x-component of gradient for ParaboloidMatrix size of [NxM-1]
			m_dyPM;           	//!< y-omponents of gradient for ParaboloidMatrix size od [N-1xM]
        Eigen::Matrix2Xd m_InterestingPoints;  	//!< Found points
        int m_InitialDeviation;           	//!< Multiplier for deviation value, can be set at constructor, default is 1
	double m_GradientInfluence;		//!< Multiplier for gradient component of deviation value, default is 1.0
	double m_AllowableError;		//!< In cross-section z = value finding there is z = value+-AllowableError is found in fact
	int m_PMcols, 				//!< The number of columns of ParaboloidMatrix, computed in constructor, can't be changed after
	    m_PMrows;				//!< The number of rows of ParaboloidMatrix, computed in constructor, can't be changed after

};

#endif /* GRIDFILTER_H */
