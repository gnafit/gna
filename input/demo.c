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


void fittest()
{
    TH1* hh = makeTH1() ;


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
    RooRealVar tau1("tau1","tau1",40,0.1,400) ;  // noLAB_th
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
    gauss.plotOn(frame,LineStyle(kDashed)) ;


    // Draw frame on canvas
    new TCanvas("energy leak fit","",600,600) ;
    gPad->SetLeftMargin(0.15) ; frame->GetYaxis()->SetTitleOffset(1.4) ;
    frame->Draw();

//*/
    }

TH1* makeTH1() 
{
    TFile fa("npe-r.root","read");
    TTree *outt=(TTree *)fa.Get("outt");
    float rmax=17000;
    float rmin=16900;
    TCut a=Form("sim_r<%f&&sim_r>%f",rmax,rmin);
    outt->Draw("sim_nPE>>htemp",a);
    TH1F *htemp=(TH1F *)gDirectory->Get("htemp");
    htemp->SetDirectory(0);
    cout<<htemp->GetEntries()<<endl;
    return htemp;
}


