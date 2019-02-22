#ifndef MINE_H 
#define MINE_H

#include <cmath>

#include <vector>
#include <map>
#include <fstream>
#include <sstream>
#include <string>
#include <math.h>
#include "TFile.h"
#include "TCanvas.h"
#include "TH1F.h"
#include "TMath.h"
#include "TGraph.h"
#include "GNAObject.hh"
#include "Eigen/Dense"
#include <algorithm>
#include <iostream>

class Mine: public GNAObject,
    public Transformation<Mine> {
        public:
            Mine();

            void getNewBins(Args args, Rets rets); 
            void normNewBins(Args args, Rets rets); 
            void setnorm(double normF){_normF =  normF;}
        protected:
            variable<double> m_p0, m_p1, m_p2, m_p3;
            double _normF;
    };

#endif // MINE_H
