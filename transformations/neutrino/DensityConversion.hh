#pragma once
#include "Units.hh"

namespace NeutrinoUnits::conversion {
          struct element {
                double per; // percentage
                int Z; // number of protons
                double A; // atomic mass
            };

          constexpr double compute_electron_fraction() {
              constexpr auto num_elems = 10;
              // Data from:
              // https://courses.lumenlearning.com/geology/chapter/reading-abundance-of-elements-in-earths-crust/
              // 10 most abundant elements are used. All the further elements are added to the Hydrogen (the last one).
              // This assumes, that all the other elements has Nn=Z
              constexpr std::array<element, num_elems> elems_in_crust =
                {{{0.460, 8, 15.999}, {0.277, 14, 28.086}, {0.081, 13, 26.982},
                 {0.05, 26, 55.847}, {0.036, 20, 40.078}, {0.028, 11, 22.990},
                 {0.026, 19, 39.098}, {0.021, 12, 24.305}, {0.004, 22, 47.867}, {0.017, 1, 1.008}} };

            // effective mass of piece of crust in MeV
              constexpr double mass_crust_eff_aem = [&](){
                  double tmp = 0.;
                  for(int j = 0; j < num_elems; j++)
                    tmp += elems_in_crust[j].per*elems_in_crust[j].A;
                  return tmp;}();

            // effective number of electrons in mass_crust_eff
              constexpr double Ne_eff = [&](){
                  double tmp = 0.;
                  for(int j = 0; j < num_elems; j++)
                    tmp += elems_in_crust[j].per*elems_in_crust[j].Z;
                  return tmp;}();

              return Ne_eff/mass_crust_eff_aem;
        }

        inline static constexpr double electron_fraction=compute_electron_fraction();
        inline static constexpr double density_to_MeV=electron_fraction/aem*g/cm3;
}
