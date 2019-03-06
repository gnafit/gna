#ifndef dybGammaPeak_h
#define dybGammaPeak_h

#include <vector>
#include <map>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <math.h>
#include <TFile.h>
#include <TCanvas.h>
#include <TH1F.h>
#include <TMath.h>
#include <TString.h>
#include <TGraph.h>
#include <TRandom.h>
#include <TProfile.h>
#include <TTree.h>
#include <TFile.h>
#include <TGraphErrors.h>
#include <Minuit2/FCNBase.h>
//#include "dybData.h"

using namespace std;

//class dybParameters;

class dybGammaPeak
{
	public:
		dybGammaPeak();
		~dybGammaPeak();

		void  Init();
		void   SetEScale     (double gamScale);
		void   SetERec       (double val);
		void   SetERecError  (double val){m_eRecError = val;}
		void   SetBiasOS     (double val){m_biasOS    = val;}
		void   GetsimDataNL  ();
		double* ScintillatorNL(double inE);
		void UpdateVis511();
		double GetChi2       ();
		bool Execute();
		double CalLh(double,double,double,double);
		void updatefQ();

		static double s_gamScale;
		static double s_reflectivity;
		string GetName       () {return m_name;       }
		double GetERec       () {return m_eRec     ;  }
		double GetEVis       () {return m_eVis     ;  }
		double GetERecError  () {return sqrt(pow(m_eRecError,2)+pow(m_eVisError,2));  }
		double GetETruSingle () {return m_eTru_single;}
		double GetETruTotal  () {return m_eTru_total; }
		double GetDataScintNL() {return m_dataScintNL;}
		double GetDataFullNL () {return m_dataFullNL; }
		double GetTheoScintNL() {return m_theoScintNL;}
		double GetTheoFullNL () {return m_theoFullNL; }
		double GetEffectiveEnergy();
		bool   IsSingleGamma(){if(fabs(m_eTru_single-m_eTru_total)<0.1) return true; else return false;}

	public:
		class MyFCN: public ROOT::Minuit2::FCNBase
	{
		public:
			MyFCN(dybGammaPeak* seek) {m_seek = seek;}
			double operator() (const std::vector<double>& x)const
			{return m_seek->CalLh(x[0],x[1],x[2],x[3]);}
			double Up() const {return 0.5;}
		private:
			dybGammaPeak *m_seek;

	};

	private:
		TGraph* m_fQ_gr;
		static int s_count;
		string m_name;
		double m_eTru_single;
		double m_eTru_total;
		double m_eVis;
		double m_gVis;
		double m_cal_Vis511[50000];
		double m_data_ekin[50000];


		float  m_predict1_NL[50000];
		float  m_predict2_NL[50000];
		float  m_data1_NL[50000];
		float  m_data2_NL[50000];

		float  sm_predict1_NL;
		float  sm_predict2_NL;
		float  sm_data1_NL;
		float  sm_data2_NL;


		int m_Ne_511[50000];
		double m_e_from511[50000][50];


		double m_eVisError;
		double m_eRec;
		double m_eRecError;
		double m_eRecRaw;
		double m_error;
		double m_biasOS;
		double m_theoScintNL;
		double m_dataScintNL;
		double m_theoFullNL;
		double m_dataFullNL;
		bool   m_includeInFit;

		/// PDF of primary e+/e- to fold with electron NL
		static const unsigned int m_nMaxPdf = 100;
		unsigned int m_nPdf;
		double        m_pdf_eTru [m_nMaxPdf];
		double        m_pdf_prob [m_nMaxPdf];
		double        m_pdf_prob2[m_nMaxPdf];
		double        m_pdf_prob3[m_nMaxPdf];
		double        m_pdf_prob4[m_nMaxPdf];
		//double        m_pdf_prob5[m_nMaxPdf];

		double Energy[1200],QuenchedE[1200],DEDX[1200];
		double ProtonE ,temp5;
		double m_kB = 6.5e-3, m_kC = 1.5e-6; // g/cm^2/MeV
		double m_kChe = 1;
		double m_Norm= 1;


		TRandom t11;
		double queE=0;
		double xx[2000];
		double yy[2000];
		int m_nnn=0;

		TProfile *m_fchren;
		TFile* fout;
		TTree* tout;
};

#endif
