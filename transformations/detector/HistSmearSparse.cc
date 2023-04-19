#include "HistSmearSparse.hh"
#include "TypesFunctions.hh"
#include <fmt/format.h>
#include <string.h>

using GNA::DataPropagation;
using GNA::MatrixType;

HistSmearSparse::HistSmearSparse(GNA::DataPropagation propagate_matrix,
                                 GNA::MatrixType matrix_type,
                                 const std::string& transformationname,
                                 const std::string& inputname,
                                 const std::string& outputname) :
GNAObjectBind1N<double>(transformationname, inputname, outputname, 1, 1, 0),
m_propagate_matrix(propagate_matrix==DataPropagation::Propagate),
m_square(matrix_type==MatrixType::Square)
{
}

TransformationDescriptor HistSmearSparse::add_transformation(const std::string& name){
  transformation_(new_transformation_name(name))
       .input("FakeMatrix")
       .dont_subscribe()
       .types(TypesFunctions::if2d<0>)
       .types(TypesFunctions::ifSameInRange<1,-1,true>)
       .types(&HistSmearSparse::types)
       .func(&HistSmearSparse::calcSmear);

  reset_open_input();
  bind_tfirst_tlast(0, 0);

  return transformations.back();
}

void HistSmearSparse::types(TypesFunctionArgs& fargs) {
  auto& args = fargs.args;
  auto& mat = args[0];
  auto& vec = args[1];

  if(m_square && mat.shape[0]!=mat.shape[1]){
    throw args.error(vec, "Input matrix is not square");
  }

  if( vec.shape[0]!=mat.shape[1] ) {
    printf("mat: \n");
    mat.dump();
    printf("vec: \n");
    vec.dump();
    throw args.error(vec, "Inputs are not multiplicable");
  }

  auto& arg1 = args[1];
  if(!arg1.defined()) return;

  auto& rets = fargs.rets;
  for(size_t i{0}; i<rets.size(); i++){
    if(m_square){
      rets[i] = arg1;
    }
    else{
      auto outedges = getOutputEdges();
      if(outedges.empty()){
        rets[i].points().shape(mat.shape[0]);
      }
      else{
        rets[i].hist().edges(outedges);
      }
    }
  }
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



