#include "HistSmearSparse.hh"
#include "TypesFunctions.hh"
#include <fmt/format.h>
#include <string.h>


HistSmearSparse::HistSmearSparse(bool propagate_matrix) :
GNAObjectBind1N("smear", "Ntrue", "Nrec", 1, 1, 0),
m_propagate_matrix(propagate_matrix)
{
}

TransformationDescriptor HistSmearSparse::add_transformation(const std::string& name){
  transformation_(new_transformation_name(name))
    .input("FakeMatrix")
    .dont_subscribe()
    .types(TypesFunctions::if2d<0>, TypesFunctions::ifSquare<0>)
    .types(TypesFunctions::ifHist<1>, TypesFunctions::ifSameInRange<1,-1,true>, TypesFunctions::passToRange<1,0,-1,true>)
    .types([](TypesFunctionArgs& fargs){
           auto& args = fargs.args;
           auto& mat = args[0];
           auto& vec = args[1];
           if( vec.shape[0]!=mat.shape[0] ) {
             throw args.error(vec, "Inputs are not multiplicable");
           }
           })
    .func(&HistSmearSparse::calcSmear);

  bind_tfirst_tlast(0, 0);

  return transformations.back();
}

/* Apply precalculated cache and actually smear */
void HistSmearSparse::calcSmear(FunctionArgs& fargs) {
  auto& args=fargs.args;
  auto& rets=fargs.rets;
  args[0]; // needed to trigger the matrix update

  for (size_t i = 0; i < rets.size(); ++i) {
    rets[i].x = m_sparse_cache * args[i+1].vec;
  }
}



