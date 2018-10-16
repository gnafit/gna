#include "HistSmearSparse.hh"
#include "TypesFunctions.hh"
#include <fmt/format.h>
#include <string.h>


HistSmearSparse::HistSmearSparse(bool single) :
m_single(single)
{
  if (single) {
    add_smear();
  }
}

TransformationDescriptor HistSmearSparse::add_smear(SingleOutput& matrix){
  auto ret=add_smear();
  ret.inputs[1].connect(matrix.single());
  return ret;
}

TransformationDescriptor HistSmearSparse::add_smear(){
  int index=static_cast<int>(transformations.size());
  std::string label="smear";
  if(!m_single){
    label = fmt::format("smear_{0}", index);
  }
  auto init=transformation_(label)
    .input("Ntrue")
    .input("FakeMatrix")
    .output("Nvis")
    .dont_subscribe()
    .types(TypesFunctions::pass<0>, TypesFunctions::ifHist<0>, TypesFunctions::if1d<0>)
    .types(TypesFunctions::if2d<1>, TypesFunctions::ifSquare<1>,
           [](TypesFunctionArgs& fargs){
           auto& args = fargs.args;
           auto& vec = args[0];
           auto& mat = args[1];
           if( vec.shape[0]!=mat.shape[0] ) {
             throw args.error(vec, "Inputs are not multiplicable");
           }
           })
    .func(&HistSmearSparse::calcSmear);

  return transformations.back();
}

/* Apply precalculated cache and actually smear */
void HistSmearSparse::calcSmear(FunctionArgs& fargs) {
  auto& args=fargs.args;
  args[1]; // needed to trigger the matrix update
  fargs.rets[0].x = m_sparse_cache * fargs.args[0].vec;
}



