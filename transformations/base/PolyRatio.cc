#include "PolyRatio.hh"
#include "TypesFunctions.hh"

template<typename FloatType>
GNA::GNAObjectTemplates::PolyRatioT<FloatType>::PolyRatioT(const std::vector<std::string> &nominator, const std::vector<std::string> &denominator) :
m_weights_nominator(nominator.size()),
m_weights_denominator(denominator.size())
{
    this->transformation_("polyratio")
        .input("points")
        .output("ratio")
        .types(TypesFunctions::pass<0>)
        .func(&PolyRatioType::polyratio);

    for (size_t i = 0; i < nominator.size(); ++i) {
        const auto& name=nominator[i];
        this->variable_(&m_weights_nominator[i], name);
    }
    for (size_t i = 0; i < denominator.size(); ++i) {
        const auto& name=denominator[i];
        this->variable_(&m_weights_denominator[i], name);
    }
    if(m_weights_nominator.size()<2lu && m_weights_denominator.size()<2lu){
        throw std::runtime_error("At least one side should be at least a first order polynomial");
    }
}

template<typename FloatType>
void GNA::GNAObjectTemplates::PolyRatioT<FloatType>::polyratio(PolyRatioT<FloatType>::FunctionArgs& fargs){
    auto& arg=fargs.args[0].x;
    auto& ret=fargs.rets[0].x;

    size_t n_nom=m_weights_nominator.size();
    size_t n_denom=m_weights_denominator.size();
    size_t niter = std::max(n_nom, n_denom);

    auto cpower=arg;
    typename Data<FloatType>::ArrayType nominator(arg.size()), denominator(arg.size());
    nominator=n_nom ? m_weights_nominator.front().value() : 1.0;
    denominator=n_denom ? m_weights_denominator.front().value() : 1.0;
    for (size_t i = 1; i < niter; ++i) {
        if(i>1){
            cpower*=arg;
        }
        if(i<n_nom){
            auto val=m_weights_nominator[i].value();
            if(val){
                nominator += cpower*val;
            }
        }
        if(i<n_denom){
            auto val=m_weights_denominator[i].value();
            if(val){
                denominator += cpower*val;
            }
        }
    }
    ret=nominator/denominator;
}

template class GNA::GNAObjectTemplates::PolyRatioT<double>;
#ifdef PROVIDE_SINGLE_PRECISION
  template class GNA::GNAObjectTemplates::PolyRatioT<float>;
#endif
