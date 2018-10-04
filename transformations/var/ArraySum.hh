#pragma once 

#include "GNAObject.hh"
#include <vector>
#include <string>

class ArraySum: public GNAObject,
                public TransformationBind<ArraySum> {
public:
      ArraySum(std::vector<std::string> names, std::string output_name);

      void exposeEvaluable();
private:
      std::vector<std::string> m_names;
      std::vector<variable<double>> m_vars;
      std::string m_output_name;
      std::vector<changeable> m_deps;
      dependant<double> m_accumulated;
};
