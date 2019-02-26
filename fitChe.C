double fitfunc(double* x, double* par)
{

    double E0=0.165;
    if (x[0]<E0) return 0.0;
    double xx=TMath::Log(1+x[0]/E0);
    return (par[0]+par[1]*xx+par[2]*pow(xx,2)+par[3]*pow(xx,3))*(1+par[4]*x[0]);



}


void fitChe(){


    TFile *fin=new TFile("mychren.root","read");
    TH1F* chren =(TH1F *)fin->Get("chren");
    TF1 *func1 = new TF1("func1", fitfunc, 0.0, 0.3, 5);
    chren->Fit(func1);
    chren->GetXaxis()->SetTitle("Kinetic Energy :[MeV]");
    chren->GetYaxis()->SetTitle("N_{Cherenkov}");
    func1->Print();

    TLegend *legend=new TLegend(0.5,0.5,0.8,0.8,"");
    Char_t message[80];
    sprintf(message,": #chi^{2}/NDF = %.2f",func1->GetChisquare()/func1->GetNDF());
    legend->AddEntry(chren,"MC Truth","lepf");
    legend->AddEntry(func1,Form("Fit function%s",message),"l");
    legend->AddEntry((TObject*)0,"(A_{0}+A_{1}.x+A_{2}.x^{2}+X_{3}.x^{3})(1+A_{4}.E)","");
    legend->AddEntry((TObject*)0,"x=ln(1+#frac{E}{E_{0}})","");
    legend->AddEntry((TObject*)0,"E_{0}=0.165 MeV","");
//    legend->AddEntry(func1,message);
    TList *liste=gPad->GetListOfPrimitives();
    cout << "\nList of objects in gPad before creating the TLegend:" << endl;
    liste->ls();
    for(Int_t i=0;i<5;i++)
    {
        func1->SetParName(i,Form("A_{%d}",i));
        sprintf(message,"%s = %.2f",
                func1->GetParName(i),func1->GetParameter(i));
        legend->AddEntry((TObject*)0,message,"");
    }
    legend->Draw();



}
