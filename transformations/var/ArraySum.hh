#pragma once 

#include "GNAObject.hh"
#include <vector>
#include <string>

class ArraySum: public GNAObject,
                public TransformationBind<ArraySum> {
public:
      ArraySum();
};
    

class SumToEvaluable: public GNAObject,
                      public TransformationBind<SumToEvaluable> {
public:
    SumToEvaluable(const std::vector<double>& arr, std::string name);
private:
    std::vector<double> m_arr;
    std::string m_name;
    dependant<double> m_accumulated;
};

