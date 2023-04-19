#include "SamplerRect.hh"

#include <iterator>
using namespace std;

int SamplerRect::offset(std::string mode){
    if(mode=="left")        return -1;
    else if(mode=="center") return 0;
    else if(mode=="right")  return 1;

    throw std::runtime_error("invalid rectangular integration mode");
    return 0;
}

void SamplerRect::fill(int offset, size_t order, double a, double b, double* abscissa, double* weight){
    if(!order){
        return;
    }
    auto binwidth=b-a;
    auto samplewidth=binwidth/order;

    double low=0, high=0;
    switch(offset){
        case -1:
            low=a;
            high=b-samplewidth;
            break;
        case 0:
            {
                double offsetwidth=samplewidth*0.5;
                low=a+offsetwidth;
                high=b-offsetwidth;
            }
            break;
        case 1:
            low=a+samplewidth;
            high=b;
            break;
        default:
            throw runtime_error("invalid offset");
    }

    switch(order){
        case 1:
            *abscissa=low;
            *weight=binwidth;
            break;
        default:
            double step = (high - low)/(order-1);
            for (size_t i = 0; i<order; ++i) {
                *abscissa=low + i*step;
                *weight=samplewidth;
                advance(abscissa,1);
                advance(weight,1);
            }
    }
}

void SamplerRect::fill_bins(int offset, size_t nbins, int* order, double* edges, double* abscissa, double* weight){
    auto* edge_a=edges;
    auto* edge_b=next(edges);
    for (size_t i=0; i < nbins; ++i) {
        size_t n = static_cast<size_t>(*order);
        fill(offset, n, *edge_a, *edge_b, abscissa, weight);
        advance(order, 1);
        advance(abscissa, n);
        advance(weight, n);
        advance(edge_a, 1);
        advance(edge_b, 1);
    }
}

