#ifndef PRODUCT_H
#define PRODUCT_H

#include "GNAObject.hh"

class Product: public GNAObject,
               public Transformation<Product> {
public:
  TransformationDef(Product)
  Product();

  void add(const OutputDescriptor &data);
  size_t size() const;

  ClassDef(Product, 1);
};

#endif // PRODUCT_H
