#pragma once
#include "GNAObject.hh"
#include <vector>
#include "UncertainParameter.hh"

/**
 * @brief Provide an array with paramete centers
 *
 * @author Maxim Gonchar
 * @date 2022.09
 */
class ParCenters: public GNAObject,
                    public TransformationBind<ParCenters> {
    public:
        ParCenters();

        ParCenters(std::vector<GaussianParameter<double>*> pars);

        void append(GaussianParameter<double>* par) {m_pars.push_back(par);};
        void materialize();

    private:
        std::vector<GaussianParameter<double>*> m_pars;
        void FillArray(FunctionArgs fargs);
};

