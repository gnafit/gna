#pragma once

#include <cmath>

#include "GNAObject.hh"
#include "TypesFunctions.hh"
#include <algorithm>

class EtoNpe: public GNAObject,
    public TransformationBind<EtoNpe> {
        public:
            EtoNpe() {
                this->transformation_("etonpe")
                    .input("E")
                    .output("npe")
                    .types(TypesFunctions::pass<0,0>)
                    .func(&EtoNpe::calcRate)
                    ;
            }

            void calcRate(FunctionArgs& fargs) {
                const auto &E = fargs.args[0].arr;
                fargs.rets[0].arr = 1348.55 + 30.71*E-4.49097*E*E+0.216868*E*E*E;
            }
    };
