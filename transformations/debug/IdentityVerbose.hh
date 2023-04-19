#pragma once

#include "GNAObject.hh"
#include "GNAObjectBind1N.hh"

//
// Identity transformation
//
class IdentityVerbose: public GNAObjectBind1N<double>,
                       public TransformationBind<IdentityVerbose> {
public:
  IdentityVerbose(std::string prefix);

  TransformationDescriptor add_transformation(const std::string& name="");

private:
  void perform(FunctionArgs& fargs);
  std::string m_prefix="";
};
