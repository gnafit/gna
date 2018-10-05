#pragma once 

#include "GNAObject.hh"
#include <vector>
#include <string>

class ArraySum: public GNAObject,
                public TransformationBind<ArraySum> {
public:
      ArraySum(SingleOutput& out);
private:
      std::string m_output_name;
      std::vector<changeable> m_deps;
      dependant<double> m_accumulated;
};
