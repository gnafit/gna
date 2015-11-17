#ifndef SUM_H
#define SUM_H

#include "GNAObject.hh"

class Sum: public GNASingleObject,
           public Transformation<Sum> {
public:
  Sum();

  void add(SingleOutput &data);
};

#endif // SUM_H
