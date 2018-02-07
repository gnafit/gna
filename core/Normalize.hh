#ifndef NORMALIZE_H
#define NORMALIZE_H

#include "GNAObject.hh"
#include "Statistic.hh"

class Normalize: public GNAObject,
                 public Transformation<Normalize> {
public:
    Normalize();
    Normalize(size_t start, size_t length);

    void doNormalize(Args args, Rets rets);
    void doNormalize_segment(Args args, Rets rets);

protected:
    void checkLimits(Atypes args, Rtypes rets);

    size_t m_start;
    size_t m_length;
};

#endif
