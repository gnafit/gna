#ifndef __CINT__
#include "RooGlobalFunc.h"
#endif
#include "RooRealVar.h"
#include "RooDataSet.h"
#include "RooGaussian.h"
#include "RooLandau.h"
#include "RooFFTConvPdf.h"
#include "RooPlot.h"
#include "TCanvas.h"
#include "TAxis.h"
#include "TH1.h"
using namespace RooFit ;


double rmin[100];
double rmax[100];

void fittest(int layers){
double rmean[100];
double mean[100];
double xer[100];
double mean1[100];
double mean2[100];
    rmin[0]=0;
    for(int ia=0;ia<layers;ia++){
        rmax[ia]=pow((ia+1.0)/layers*17200*17200*17200,1.0/3.0);
        if(ia!=0)rmin[ia]=rmax[ia-1];
        cout<<rmin[ia]<<" "<<rmax[ia]<<endl;
    }

    double *meantmp;
    double cc[100];
    ifstream inf("mmm2");
    char buffer[500];
    for(int ia=0;ia<layers;ia++){
        inf.getline(buffer,sizeof(buffer));
        istringstream(buffer)>>mean[ia]>>mean1[ia]>>mean2[ia];
        rmean[ia]=0.5*(rmin[ia]+rmax[ia]);
        xer[ia]=0.5*(-rmin[ia]+rmax[ia]);
        if(ia>69&&ia<100)
        {
        meantmp=fittest2(ia);
        mean[ia]=meantmp[0];
        mean1[ia]=meantmp[1];
        mean2[ia]=meantmp[2];
        }
        cc[ia]=ia;
    }
    for(int ia=0;ia<layers;ia++){
        cout<<mean[ia]<<"  "<<mean1[ia]<<" "<<mean2[ia]<<endl;
    }
    TCanvas *mm2=new TCanvas("mm2","",800,600);
    gr = new TGraphAsymmErrors(layers, rmean,mean,xer,xer,mean1,mean2);
    gr->Draw("ap");
    TCanvas *mm=new TCanvas("mm","",800,600);
    gr2 = new TGraph(layers,cc,mean);
    gr2->Draw("ap");

}
double* fittest2(int nnj)
{
    TH1* hh = makeTH1(nnj) ;


    // S e t u p   c o m p o n e n t   p d f s 
    // ---------------------------------------

    // Construct observable
    RooRealVar sim_nPE("sim_nPE","sim_nPE",800,1800) ;

    RooDataHist dh("dh","dh",RooArgSet(sim_nPE),Import(*hh)) ;
    dh.Print();

    //   /*

    // Construct landau(t,ml,sl) ;
    RooRealVar mg("mg","mg",1300,1100,1550) ;
    RooRealVar sg("sg","sg",50.,10,80) ;
    RooGaussModel gauss("gauss","gauss",sim_nPE,mg,sg) ;

    // low energy tail 
    RooRealVar tau1("tau1","tau1",40,0.0001,400) ;  // noLAB_th
    RooDecay decay("decay","decay",sim_nPE,tau1,gauss,RooDecay::Flipped ) ;

    // C o n s t r u c t   c o n v o l u t i o n   p d f 
    // ---------------------------------------

    // Set #bins to be used for FFT sampling to 10000
    //sim_nPE.setBins(1000,"cache") ; 




    // S a m p l e ,   f i t   a n d   p l o t   c o n v o l u t e d   p d f 
    // ----------------------------------------------------------------------

    // Sample 1000 events in x from gxlx
    //RooDataSet* data = decay.generate(t,30000) ;

    // Fit gxlx to data
    //decay.fitTo(*data) ;
    decay.fitTo(dh) ;

    // Plot data, landau pdf, landau (X) gauss pdf
    RooPlot* frame = sim_nPE.frame(Title("fit low energy tail")) ;
    dh->plotOn(frame) ;
    decay.plotOn(frame) ;
    gauss.plotOn(frame,LineStyle(kDashed),LineColor(kMagenta)) ;


    // Draw frame on canvas
    new TCanvas("energy leak fit","",600,600) ;
    gPad->SetLeftMargin(0.15) ; frame->GetYaxis()->SetTitleOffset(1.4) ;
    TCanvas c;
    frame->Draw();
    c.SaveAs(Form("pp_%d.C",nnj));
    c.SaveAs(Form("pp_%d.png",nnj));
    double *rer=new double[3];
    int ia=0;
    rer[ia++]=mg->getVal();
    rer[ia++]=mg->getErrorHi();
    rer[ia++]=mg->getErrorLo();
    return rer;

    //*/
}

TH1* makeTH1(int nnn) 
{
    TFile fa("npe-r.root","read");
    TTree *outt=(TTree *)fa.Get("outt");
    TCut a=Form("sim_r<%f&&sim_r>%f",rmax[nnn],rmin[nnn]);
    outt->Draw("sim_nPE>>htemp",a);
    TH1F *htemp=(TH1F *)gDirectory->Get("htemp");
    htemp->SetDirectory(0);
    cout<<htemp->GetEntries()<<endl;
    return htemp;
}


