#pragma once

#include "GNAObject.hh"

class Product: public GNASingleObject,
               public TransformationBind<Product> {
public:
  Product();

  void multiply(SingleOutput &data);
  size_t size() const;
};
