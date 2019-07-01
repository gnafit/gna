#include "HistEdgesLinear.hh"
#include "TypeClasses.hh"
#include "fmt/format.h"

#include <algorithm>

void HistEdgesLinear::init(){
    auto trans = this->transformation_("histedges");
    trans.input("hist_in", /*inactive*/true)// 0 - histogram with bins defined
         .output("hist")                    // 2 - histogram with new edges
         .types(new TypeClasses::CheckKindT<double>(DataKind::Hist), new TypeClasses::CheckNdimT<double>(1))
         .types(&HistEdgesLinear::types)
         .func(&HistEdgesLinear::func);
         ;
}

void HistEdgesLinear::types(TypesFunctionArgs& fargs){
    auto& hist_input = fargs.args[0];
    auto& hist_output = fargs.rets[0];
    hist_output = hist_input;
    auto& edges = hist_output.edges;
    std::transform(edges.begin(), edges.end(), edges.begin(), [this](double v){ return this->m_k*v + this->m_b; });
}

void HistEdgesLinear::func(FunctionArgs& fargs){
    auto& hist = fargs.rets[0];
    hist.x     = 0.0;
}
