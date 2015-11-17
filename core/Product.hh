#ifndef PRODUCT_H
#define PRODUCT_H

#include "GNAObject.hh"

class Product: public GNASingleObject,
               public Transformation<Product> {
public:
  Product();

  void multiply(SingleOutput &data);
  size_t size() const;
};

#endif // PRODUCT_H
