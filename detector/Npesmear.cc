#include <boost/math/constants/constants.hpp>
#include "Npesmear.hh"
#include <boost/format.hpp>
#include <string.h>
using boost::format;

constexpr double pi = boost::math::constants::pi<double>();

Npesmear::Npesmear( bool single ) : m_single(single) {
    if (single) {
        add();
    }
}

void Npesmear::add(){
    int index=static_cast<int>(transformations.size());
    std::cout<<" index "<<index<<std::endl;
    std::string label="Npesmear";
    if(!m_single){
        label=(format("Npesmear_%1%")%index).str();
    }
    transformation_(this, label)
        .input("Nvis")
        .output("Nrec")
        .types(Atypes::pass<0>,
                [index](Npesmear *obj, Atypes args, Rtypes /*rets*/) {
                std::cout<<" ypp0 "<<std::endl;
                if( args[0].kind!=DataKind::Hist ){
                throw std::runtime_error("Npesmear input should be a histogram");
                }
                if(index==0){
                obj->m_datatype = args[0];
                std::cout<<" yp0 "<<std::endl;
                obj->fillCache();
                std::cout<<" yp1 "<<std::endl;
                }
                else{
                if( args[0]!=obj->m_datatype ) {
                throw std::runtime_error("Inconsistent histogram in Npesmear");
                }
                }
                })
    .func(&Npesmear::calcSmear);
}

void Npesmear::add(SingleOutput& hist){
    if( m_single ){
        throw std::runtime_error("May have only single energy resolution transformation in this class instance");
    }
    add();
    transformations.back().inputs[0]( hist.single() );
}


double Npesmear::resolution(double dEtrue, double drec) const noexcept {

    const double Etrue = 1348.55 + 30.71*dEtrue;
    const int Erec= int(1348.55 + 30.71*drec);
    return std::exp(-Etrue)*std::pow(Etrue,Erec)/std::tgamma(Erec);
}

void Npesmear::fillCache() {
    m_size = m_datatype.hist().bins();
    std::cout<<"yp  m_size "<<m_size<<std::endl;
    if (m_size == 0) {
        return;
    }
    m_sparse_cache.resize(m_size, m_size);

    /* fill the cache matrix with probalilities for number of events to leak to other bins */
    /* colums corressponds to reconstrucred energy and rows to true energy */
    auto bin_center = [&](size_t index){ return (m_datatype.edges[index+1] + m_datatype.edges[index])/2; };
    for (size_t etrue = 0; etrue < m_size; ++etrue) {
        double Etrue = bin_center(etrue);
        double dEtrue = m_datatype.edges[etrue+1] - m_datatype.edges[etrue];

        bool right_edge_reached{false};
        /* precalculating probabilities for events in given bin to leak to
         * neighbor bins  */
        for (size_t erec = 0; erec < m_size; ++erec) {
            double Erec = bin_center(erec);
            double rEvents = dEtrue*resolution(Etrue, Erec);

            if (rEvents < 1E-10) {
                if (right_edge_reached) {
                    break;
                }
                continue;
            }
            if(erec==etrue) rEvents=1;
            if(erec!=etrue) rEvents=0;
            m_sparse_cache.insert(erec, etrue) = rEvents;
            //std::cout<<erec<<"\t"<<etrue<<"\t"<<rEvents<<std::endl;
            if (!right_edge_reached) {
                right_edge_reached = true;
            }
        }
    }
    m_sparse_cache.makeCompressed();
}

/* Apply precalculated cache and actually smear */
void Npesmear::calcSmear(Args args, Rets rets) {
    // rets[0].x = args[0].vec ;
    rets[0].x = m_sparse_cache * args[0].vec ;
    rets[0].x = 1348.55 + 30.71*rets[0].x;//-4.49097*E*E+0.216868*E*E*E;
}



