#pragma once

#include "GNAObject.hh"
#include <vector>
#include <string>

class ArraySum: public GNAObject,
                public TransformationBind<ArraySum> {
public:
      ArraySum()= default;
      ArraySum(const std::string& name, SingleOutput& out);

private:
      void initialize(const std::string& name);
      void sum(FunctionArgs& fargs);
      void check(TypesFunctionArgs& fargs);

      std::string m_output_name;
      dependant<double> m_accumulated;
};
