#ifndef PRODUCT_H
#define PRODUCT_H

#include "GNAObject.hh"

class Product: public GNAObject,
               public Transformation<Product> {
public:
  TransformationDef(Product)
  Product();

  void multiply(const TransformationDescriptor &obj) {
    multiply(obj.outputs.single());
  }
  void multiply(const TransformationDescriptor::Outputs &outs) {
    multiply(outs.single());
  }
  void multiply(const OutputDescriptor &data);
  size_t size() const;

  ClassDef(Product, 1);
};

#endif // PRODUCT_H
