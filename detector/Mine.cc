#include "Mine.hh"
#include "tt.hh"



Mine::Mine(){
	variable_(&m_p0, "Qp0");
	variable_(&m_p1, "Qp1");
	variable_(&m_p2, "Qp2");
	variable_(&m_p3, "Qp3");

	transformation_(this, "MineNL")
		.input("old_bins")
		.output("bins_after_nl")
		.types(Atypes::pass<0,0>)
		.func(&Mine::getNewBins)
		;
	 transformation_(this, "DisplayNL")
	     .input("old_bins2")
	     .output("bins2_after_nl")
	     .types(Atypes::pass<0,0>)
	     .func(&Mine::fordisplay)
	     ;
	transformation_(this, "normMineNL")
		.input("new_bins")
		.output("norm_new_bins")
		.types(Atypes::pass<0,0>)
		.func(&Mine::normNewBins)
		;

}

void Mine::normNewBins( Args args, Rets rets){
	const auto& orig_bin_edges = args[0].arr;
	rets[0].arr = _normF*orig_bin_edges;
}
// /*
   void Mine::fordisplay(Args args, Rets rets){
   double OrigEnergy[1200],DEDX[1200];
   std::ifstream in2("stoppingpower.txt");
   for(int i=0;i<1200;i++){
   std::string temp2;
   double temp3;
   in2>>OrigEnergy[i]>>temp2>>temp3>>DEDX[i];
   }




int m_totalpoints = 0;
double *xx=new double[2000];
double *yy=new double[2000];
for(int ii=0;ii<1200;ii++)
{

double energy_point=OrigEnergy[ii];

double queE=0;

for(int i=ii;;i--){
if(energy_point<0.005) break;
double dedx = DEDX[i];

double Q = 1+dedx*m_p0+dedx*dedx*m_p1;
queE+=0.01/Q;
energy_point-=0.01;
}
xx[m_totalpoints]=OrigEnergy[ii];
yy[m_totalpoints]=queE;
m_totalpoints++;
}
std::vector<double> quch_Yv(&yy[0],&yy[0]+m_totalpoints);
Eigen::VectorXd quch_Yvals = Eigen::Map<Eigen::VectorXd, Eigen::Unaligned>(quch_Yv.data(), quch_Yv.size());
std::vector<double> quch_Xv(&xx[0],&xx[0]+m_totalpoints);
Eigen::VectorXd quch_Xvals = Eigen::Map<Eigen::VectorXd, Eigen::Unaligned>(quch_Xv.data(), quch_Xv.size());

const auto& orig_bin_edges = args[0].arr-2*0.511;
SplineFunction quch_s(quch_Xvals, quch_Yvals);
std::cout<<" m_p0: "<<m_p0<<" m_p1: "<<m_p1;
std::cout<<" m_p2: "<<m_p2<<" m_p3: "<<m_p3<< std::endl;
//rets[0].arr = (m_p2*quch_s.eval(2*0.511)+ m_p2*quch_s.eval(orig_bin_edges)+m_p3*cheren_s.eval(orig_bin_edges))/1234./(orig_bin_edges+2*0.511);
	double E0=0.165;
	double par[5]={ -7.26624e+00, 1.72463e+01, -2.18044e+01,1.44731e+01,3.22121e-02};
rets[0].arr = (m_p2*quch_s.eval(2*0.511) + m_p2*quch_s.eval(orig_bin_edges)+m_p3*(par[0]+par[1]*log(1+orig_bin_edges/E0)+par[2]*pow(log(1+orig_bin_edges/E0),2)+par[3]*pow(log(1+orig_bin_edges/E0),3))*(1+par[4]*orig_bin_edges))/1234.0/(orig_bin_edges+2*0.511);
//rets[0].arr = (1261 + m_p2*quch_s.eval(orig_bin_edges)+m_p3*(par[0]+par[1]*log(1+orig_bin_edges/E0)+par[2]*pow(log(1+orig_bin_edges/E0),2)+par[3]*pow(log(1+orig_bin_edges/E0),3))*(1+par[4]*orig_bin_edges))/1234.0/(orig_bin_edges+2*0.511);
}
// */
void Mine::getNewBins(Args args, Rets rets){
	double OrigEnergy[1200],DEDX[1200];
	TH1F *m_Cheren;
	std::ifstream in2("stoppingpower.txt");
	for(int i=0;i<1200;i++){
		std::string temp2;
		double temp3;
		in2>>OrigEnergy[i]>>temp2>>temp3>>DEDX[i];
	}

	TFile fileCheren("./mychren.root","read");
	m_Cheren=(TH1F *)fileCheren.Get("chren");
	//cout<<"cheng "<<chren->Interpolate(4)<<endl;
	m_Cheren->SetDirectory(0);

	int Nbinarr=m_Cheren->GetNbinsX();
	float* CYarr=m_Cheren->GetArray();
	std::vector<float> CYv(&CYarr[1],&CYarr[1]+Nbinarr);
	delete CYarr;
	std::vector<double> cheren_Vec(CYv.begin(), CYv.end());
	Eigen::VectorXd cheren_xvals = Eigen::VectorXd::LinSpaced(1000, 0.005000, 9.995000);
	Eigen::VectorXd cheren_yvals = Eigen::Map<Eigen::VectorXd, Eigen::Unaligned>(cheren_Vec.data(), cheren_Vec.size());

	//cheren_xvals << 0, 15, 30;
	//cheren_yvals << 0, 12, 17;

	//std::cout<<cheren_xvals[0]<<"\t"<<cheren_yvals[0]<<std::endl;
	//std::cout<<cheren_xvals[1]<<"\t"<<cheren_yvals[1]<<std::endl;
	//std::cout<<cheren_xvals[999]<<"\t"<<cheren_yvals[999]<<std::endl;

	//std::cout << s.eval(2.34) << std::endl;

	int m_totalpoints = 0;
	double *xx=new double[2000];
	double *yy=new double[2000];
	for(int ii=0;ii<1200;ii++)
	{

		double energy_point=OrigEnergy[ii];

		double queE=0;

		for(int i=ii;;i--){
			if(energy_point<0.005) break;
			double dedx = DEDX[i];

			double Q = 1+dedx*m_p0+dedx*dedx*m_p1;
			//cout<<i<<' '<<j<<' '<<energy_point<<' '<<dedx<<' '<<dedx1<<endl;
			queE+=0.01/Q;
			energy_point-=0.01;
		}
		xx[m_totalpoints]=OrigEnergy[ii];
		yy[m_totalpoints]=queE;
		m_totalpoints++;
	}
	std::vector<double> quch_Yv(&yy[0],&yy[0]+m_totalpoints);
	Eigen::VectorXd quch_Yvals = Eigen::Map<Eigen::VectorXd, Eigen::Unaligned>(quch_Yv.data(), quch_Yv.size());
	std::vector<double> quch_Xv(&xx[0],&xx[0]+m_totalpoints);
	Eigen::VectorXd quch_Xvals = Eigen::Map<Eigen::VectorXd, Eigen::Unaligned>(quch_Xv.data(), quch_Xv.size());

	const auto& orig_bin_edges = args[0].arr;
	//const auto& orig_bin_edges = args[0].arr-2*0.511;
	SplineFunction cheren_s(cheren_xvals, cheren_yvals);
	SplineFunction quch_s(quch_Xvals, quch_Yvals);
	std::cout<<" m_p0: "<<m_p0<<" m_p1: "<<m_p1;
	std::cout<<" m_p2: "<<m_p2<<" m_p3: "<<m_p3<< std::endl;
	//         rets[0].arr = (m_p2*quch_s.eval(2*0.511)+ m_p2*quch_s.eval(orig_bin_edges)+m_p3*cheren_s.eval(orig_bin_edges))/1234./(orig_bin_edges+2*0.511);
	//rets[0].arr = (1261 + m_p2*quch_s.eval(orig_bin_edges)+m_p3*cheren_s.eval(orig_bin_edges))/1234.0;
	// use fit cheren response
	double E0=0.165;
	double par[5]={ -7.26624e+00, 1.72463e+01, -2.18044e+01,1.44731e+01,3.22121e-02};
	//if(orig_bin_edges>E0) To Be Done, there is a cut off for Cherenkov response below E0
		rets[0].arr = (m_p2*quch_s.eval(orig_bin_edges)+m_p3*(par[0]+par[1]*log(1+orig_bin_edges/E0)+par[2]*pow(log(1+orig_bin_edges/E0),2)+par[3]*pow(log(1+orig_bin_edges/E0),3))*(1+par[4]*orig_bin_edges))/1234.0/1.12722853;
		//rets[0].arr = (m_p2*quch_s.eval(2*0.511) + m_p2*quch_s.eval(orig_bin_edges)+m_p3*(par[0]+par[1]*log(1+orig_bin_edges/E0)+par[2]*pow(log(1+orig_bin_edges/E0),2)+par[3]*pow(log(1+orig_bin_edges/E0),3))*(1+par[4]*orig_bin_edges))/1234.0;
		//rets[0].arr = (1261 + m_p2*quch_s.eval(orig_bin_edges)+m_p3*(par[0]+par[1]*log(1+orig_bin_edges/E0)+par[2]*pow(log(1+orig_bin_edges/E0),2)+par[3]*pow(log(1+orig_bin_edges/E0),3))*(1+par[4]*orig_bin_edges))/1234.0;
	//else 
	//	rets[0].arr = (1261 + m_p2*quch_s.eval(orig_bin_edges))/1234.0;
	//std::cout <<" args "<<args[0].arr<< std::endl << std::endl;
	//std::cout <<" rets "<<rets[0].arr<< std::endl << std::endl;

	//    std::cout<<"single "<< (1261 + m_p2*quch_s.eval(0)+m_p3*cheren_s.eval(0))/1234./1.022<<std::endl;
	//    std::cout<<"single "<< (1261 + m_p2*quch_s.eval(4)+m_p3*cheren_s.eval(4))/1234./(4+1.022)<<std::endl;
	//    std::cout<<"single "<< (1261 + m_p2*quch_s.eval(8)+m_p3*cheren_s.eval(8))/1234./(8+1.022)<<std::endl;
}
