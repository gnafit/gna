#pragma once

#include <cmath>

#include "GNAObject.hh"
#include "TypesFunctions.hh"
#include <algorithm>

class Worst: public GNAObject,
             public TransformationBind<Worst> {
        public:
            Worst() {
                this->transformation_("WorstNL")
                    .input("old_bins")
                    .output("bins_after_nl")
                    .types(TypesFunctions::pass<0,0>)
                    .func(&Worst::getNewBins)
                    ;
            }

            void getNewBins(FunctionArgs& fargs) {
                const auto &xx = fargs.args[0].arr+0.8;
                double ms32=0.00251;//inverted
                //double ms32=0.00244;//normal
                double th12=asin(sqrt(0.304));
                double c2th12=pow(cos(th12),2);
                double s2th12=pow(sin(th12),2);
                double ms21=0.0000753;
                double L=55000.0;
                Eigen::ArrayXd delta12=1.27*ms21*L/xx;
                Eigen::ArrayXd sphi=(c2th12*sin(2*delta12))/(sqrt(1-4*s2th12*c2th12*pow(sin(delta12),2)));
                Eigen::ArrayXd cphi=(c2th12*cos(2*delta12)+s2th12)/(sqrt(1-4*s2th12*c2th12*pow(sin(delta12),2)));
                //Eigen::ArrayXd phi = sphi.binaryExpr(cphi, std::ptr_fun(::atan2));
                //Eigen::ArrayXd phi = sphi.binaryExpr(cphi,[](double a, double  b){return std::atan2(a,b);});
                Eigen::ArrayXd phi = sphi.binaryExpr(cphi, [] (double a, double b) {
                        double tmpphi=asin(a);
                        if(a<0){if(b>0) tmpphi = tmpphi+6.28;else tmpphi = 3.14-tmpphi;}
                        if(a>0&&b<0) tmpphi=3.14-tmpphi;
                        return tmpphi;} );
                Eigen::ArrayXd msphi=phi*xx/1.27/L;
                double in1=4*c2th12*ms21;
                fargs.rets[0].arr = xx*(2*ms32+in1-msphi)/(2*ms32+msphi)-0.8;
    //std::cout <<" args "<<fargs.args[0].arr<< std::endl << std::endl;
    //std::cout <<" rets "<<fargs.rets[0].arr<< std::endl << std::endl;
            }

    };
