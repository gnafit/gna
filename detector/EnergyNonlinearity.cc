#include "EnergyNonlinearity.hh"
#include <iostream>
using namespace std;

EnergyNonlinearity::EnergyNonlinearity() {
  transformation_(this, "smear")
      .input("Matrix")
      .input("Ntrue")
      .output("Nvis")
      .types(Atypes::pass<0>,
         [](EnergyNonlinearity *obj, Atypes args, Rtypes /*rets*/) {
           //obj->fillCache();
         })
       .func(&EnergyNonlinearity::calcSmear);

  transformation_(this, "matrix")
      .input("Edges")
      .input("EdgesModified")
      .output("Matrix")
      .types(Atypes::ifSame,
         [](EnergyNonlinearity *obj, Atypes args, Rtypes rets) {
           obj->m_size = args[0].shape[0]-1;
           obj->m_sparse_cache.resize(obj->m_size, obj->m_size);
           rets[0] = obj->m_datatype = DataType().points().shape( obj->m_size, obj->m_size );
         })
       .func(&EnergyNonlinearity::calcMatrix);
}

void EnergyNonlinearity::set( SingleOutput& bin_edges, SingleOutput& bin_edges_modified, SingleOutput& ntrue ){
    if( m_initialized )
        throw std::runtime_error("EnergyNonlinearity is already initialized");
    m_initialized = true;

    t_["matrix"].inputs()[0].connect( bin_edges.single() );
    t_["matrix"].inputs()[1].connect( bin_edges_modified.single() );
    t_["smear"].inputs()[0].connect( t_["matrix"].outputs()[0] );
    t_["smear"].inputs()[1].connect( ntrue.single() );
}

void EnergyNonlinearity::calcSmear(Args args, Rets rets) {
  rets[0].x = m_sparse_cache * args[0].vec;
}

void EnergyNonlinearity::calcMatrix(Args args, Rets rets) {
  m_sparse_cache.setZero();

  auto& edges_orig=args[0];
  auto& edges_mod=args[1];
  std::vector<int> bins( edges_mod.size() );
  std::vector<double> widths( edges_mod.size()-1 );

  double* b_edges_orig=edges_orig.data();
  double* b_edges_mod=edges_m.data();
  double* b_edges_orig_end=b_edges_orig+edges_orig.size();
  double* b_edges_mod_end=b_edges_mod+edges_mod.size();

  int i_orig(0);
  for (int i_mod = 0; i_mod < edges_mod.size(); ++i_mod, ++b_edges_mod) {
    if( *b_edges_mod < *b_edges_orig ) {
      bins[i_mod]
    }


  }

  //for(int i=0; i<m_size; i++){
    //double le1 = edges_m(i);
    //double ue1 = edges_m(i+1);
    //if ( le1==0.0 || ue1==0.0 ) continue;

    //if ( le1>ue1 ) std::swap( le1, ue1 );
    //double width1 = ue1-le1;

    //int lbin = hout->FindBin( le1 );
    //int ubin = hout->FindBin( ue1 );
    //if ( ubin==0 ){ // underflow
      //continue;
    //}
    //double uei = edges( lbin )+width0;
    //double remainder = 1.0;
    //if ( lbin==m_size ) {
      //continue;
    //}
    //if ( lbin<0 ) lbin=0;

    ////printf( "Bin %i (%f, %f) to (%f, %f): ", i, hin->GetBinLowEdge(i+1), hin->GetBinLowEdge(i+2), le1, ue1 );
    //while ( remainder>1.e-12 && le1<=ue1){
      //if ( lbin==m_size ){ // overflow
        //remainder=0.0;
        //break;
      //}
      //double frac = width1>0.0 ? ( min( uei, ue1 ) - le1 )/width1 : remainder;
      //m_sparse_cache.insert(lbin,i)=frac;

      //le1=uei;
      //uei+=width0;
      //lbin++;
      //remainder-=frac;
    //}

  rets[0].mat = m_sparse_cache;
}

//void EnergyNonlinearity::fillCache() {
  //m_size = m_datatype.hist().bins();
  //if (m_size == 0) {
    //return;
  //}
  //m_sparse_cache.resize(m_size, m_size);

  //[> fill the cache matrix with probalilities for number of events to leak to other bins <]
  //[> colums corressponds to reconstrucred energy and rows to true energy <]
  //auto bin_center = [&](size_t index){ return (m_datatype.edges[index+1] + m_datatype.edges[index])/2; };
  //for (size_t etrue = 0; etrue < m_size; ++etrue) {
    //double Etrue = bin_center(etrue);
    //double dEtrue = m_datatype.edges[etrue+1] - m_datatype.edges[etrue];

    //bool right_edge_reached{false};
    /* precalculating probabilities for events in given bin to leak to
     * neighbor bins  */
    //for (size_t erec = 0; erec < m_size; ++erec) {
      //double Erec = bin_center(erec);
      //double rEvents = dEtrue*resolution(Etrue, Erec);

      //if (rEvents < 1E-10) {
        //if (right_edge_reached) {
           //break;
        //}
        //continue;
      //}
      //m_sparse_cache.insert(erec, etrue) = rEvents;
      //if (!right_edge_reached) {
        //right_edge_reached = true;
      //}
    //}
  //}
  //m_sparse_cache.makeCompressed();
//}



