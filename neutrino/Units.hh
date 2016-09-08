#ifndef UNITS_H
#define UNITS_H

// energy
static constexpr double MeV = 1.;
static constexpr double eV = 1.e-6*MeV;
static constexpr double eV2 = eV*eV;

// length
static constexpr double km  = 1.e16/1.973269718/MeV;                        
static constexpr double m  = 1.e-3*km;                        
static constexpr double ns = 1;

#endif
