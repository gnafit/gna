#include "EnergyNonlinearity.hh"

EnergyNonlinearity::EnergyNonlinearity() {
  //variable_(&m_a, "Eres_a");
  //variable_(&m_b, "Eres_b");
  //variable_(&m_c, "Eres_c");
  //callback_([this] { fillCache(); });

  transformation_(this, "smear")
      .input("Ntrue")
      .output("Nvis")
      .types(Atypes::pass<0>,
         [](EnergyNonlinearity *obj, Atypes args, Rtypes /*rets*/) {
           obj->m_datatype = args[0];
           //obj->fillCache();
         })
       .func(&EnergyNonlinearity::calcSmear);
}

/* Apply precalculated cache and actually smear */
void EnergyNonlinearity::calcSmear(Args args, Rets rets) {
  rets[0].x = m_sparse_cache * args[0].vec;
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



