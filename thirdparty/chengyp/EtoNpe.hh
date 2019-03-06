#ifndef ETONPE_H
#define ETONPE_H

#include <cmath>

#include "GNAObject.hh"
#include <algorithm>

class EtoNpe: public GNAObject,
    public Transformation<EtoNpe> {
        public:
            EtoNpe() {
                transformation_(this, "etonpe")
                    .input("E")
                    .output("npe")
                    .types(Atypes::pass<0,0>)
                    .func(&EtoNpe::calcRate)
                    ;
            }

            void calcRate(Args args, Rets rets) {
                const auto &E = args[0].arr;
                rets[0].arr = 1348.55 + 30.71*E-4.49097*E*E+0.216868*E*E*E;
            }
    };

#endif // ETONPE_H
