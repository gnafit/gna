#include "VarArray.hh"

using std::next;

VarArray::VarArray(const std::vector<std::string>& varnames)
  : m_vars(varnames.size())
{
  for (size_t i = 0; i < varnames.size(); ++i) {
    variable_(&m_vars[i], varnames[i]);
  }

  transformation_("vararray")                               /// Initialize the transformation points.
    .output("points")
    .types(&VarArray::typesFunction)
    .func(&VarArray::function)
    .finalize();                                            /// Tell the initializer that there are no more configuration and it may initialize the types.
}

void VarArray::typesFunction(Atypes args, Rtypes rets){
  rets[0] = DataType().points().shape(m_vars.size());
}

void VarArray::function(Args args, Rets rets){
  auto* buffer = rets[0].buffer;
  for( auto var : m_vars ){
      *buffer = var.value();
      buffer=next(buffer);
  }
}
