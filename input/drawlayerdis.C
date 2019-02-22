void drawlayerdis(int layers){
    double rmin[layers];
    double rmax[layers];
    rmin[0]=0;
    for(int ia=0;ia<layers;ia++){
        rmax[ia]=pow((ia+1.0)/layers*17200*17200*17200,1.0/3.0);
        if(ia!=0)rmin[ia]=rmax[ia-1];
        cout<<rmin[ia]<<" "<<rmax[ia]<<endl;
    }
    gStyle->SetOptStat(0);
    gStyle->SetOptFit(kFALSE);

    TFile fa("npe-r.root","read");
    TTree *outt=(TTree *)fa.Get("outt");
    //outt->Draw("sim_nPE","sim_r<12100&&sim_r>11900","colzh");
    TCanvas *can[layers];
    for(int ia=0;ia<layers;ia++){
        can[ia]=new TCanvas(Form("c%d",ia),Form("c%d",ia),800,600);
        can[ia]->cd();
        TCut a=Form("sim_r<%f&&sim_r>%f",rmax[ia],rmin[ia]);
        outt->Draw("rec_nPE>>htemp",a);
        TH1F *htemp=(TH1F *)gDirectory->Get("htemp");
        htemp->Fit("gaus","q");
        htemp->SetDirectory(0);
        TF1 *tf=htemp->GetFunction("gaus");
        auto rp = new TRatioPlot(htemp);
        rp->Draw();
        rp->GetLowerRefYaxis()->SetTitle("ratio");
        rp->GetUpperRefYaxis()->SetTitle("entries");

        cout<<" chi2 "<<tf->GetChisquare()/tf->GetNDF()<<" ";
        cout << tf->GetParameter(2)/((tf->GetParameter(1)))<< endl; 
        tf->Clear();
    }



}

/*
   void ratioplot1() {
   gStyle->SetOptStat(0);
   auto c1 = new TCanvas("c1", "A ratio example");
   auto h1 = new TH1D("h1", "h1", 50, 0, 10);
   auto h2 = new TH1D("h2", "h2", 50, 0, 10);
   auto f1 = new TF1("f1", "exp(- x/[0] )");
   f1->SetParameter(0, 3);
   h1->FillRandom("f1", 1900);
   h2->FillRandom("f1", 2000);
   h1->Sumw2();
   h2->Scale(1.9 / 2.);
   h1->GetXaxis()->SetTitle("x");
   h1->GetYaxis()->SetTitle("y");
   auto rp = new TRatioPlot(h1, h2);
   c1->SetTicks(0, 1);
   rp->Draw();
   c1->Update();
   }
   */
