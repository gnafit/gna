#ifndef NORMALIZE_H
#define NORMALIZE_H

#include "GNAObject.hh"
#include "Statistic.hh"

class Normalize: public GNAObject,
                 public Transformation<Normalize> {
public:
    Normalize();
    Normalize(int start, int length);

    void normalize(Args args, Rets rets);
    void normalize_segment(Args args, Rets rets);

private:
    int m_start;
    int m_length;
};

#endif
