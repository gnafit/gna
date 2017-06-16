{
  gDirectory->AddDirectory( false );
  TH1::AddDirectory( false );

  gSystem->Load("libGlobalNuAnalysis2.so");
}
