#pragma once

#include "GNAObject.hh"
#include <vector>
#include <string>

class ArraySum: public GNAObject,
                public TransformationBind<ArraySum> {
public:
      ArraySum(){};
      ArraySum(const std::string& name, SingleOutput& out);

private:
      void initialize(const std::string& name, SingleOutput& out);

      std::string m_output_name;
      dependant<double> m_accumulated;
};
