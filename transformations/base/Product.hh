#pragma once

#include "GNAObject.hh"

class Product: public GNASingleObject,
               public TransformationBind<Product> {
public:
  Product();

  void multiply(SingleOutput &data);
  void multiply(SingleOutput &data1, SingleOutput &data2){ multiply(data1); multiply(data2); }
  void multiply(SingleOutput &data1, SingleOutput &data2, SingleOutput &data3){ multiply(data1); multiply(data2, data3); }
  size_t size() const;
};
