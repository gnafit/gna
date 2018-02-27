#pragma once

#include "GNAObject.hh"

class VarArray: public GNASingleObject,
                public TransformationBind<VarArray> {
public:
  VarArray(const std::vector<std::string>& varnames); ///< Constructor.

protected:
  void typesFunction(Atypes args, Rtypes rets);
  void function(Args args, Rets rets);
  std::vector<variable<double>> m_vars;              ///< List of variables.
};
