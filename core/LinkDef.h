#ifdef __CINT__

#pragma link off all globals;
#pragma link off all classes;
#pragma link off all functions;

//
// A collection of classes
//

#pragma link C++ class KeyError-;
#pragma link C++ class IndexError-;

#pragma link C++ class TransformationDescriptor+;
#pragma link C++ class InputDescriptor+;
#pragma link C++ class OutputDescriptor+;
#pragma link C++ class TransformationDescriptor::InputsBase+;
#pragma link C++ class TransformationDescriptor::Inputs+;
#pragma link C++ class TransformationDescriptor::OutputsBase+;
#pragma link C++ class TransformationDescriptor::Outputs+;

#pragma link C++ class VariableDescriptor+;
#pragma link C++ class EvaluableDescriptor+;
#pragma link C++ class EvaluableDescriptor::Sources-;

#pragma link C++ class GNAObject;
#pragma link C++ class GNAObject::Variables-;
#pragma link C++ class GNAObject::Evaluables-;
#pragma link C++ class GNAObject::Transformations-;
#pragma link C++ class GNASingleObject;

#pragma link C++ class ExpressionsProvider-;

#pragma link C++ class ParameterWrapper<double>-;
#pragma link C++ class Uncertain<double>-;
#pragma link C++ class Uncertain<std::complex<double>>-;
#pragma link C++ class Uncertain<std::array<double,3>>-;
#pragma link C++ class Parameter<double>-;
#pragma link C++ class GaussianParameter<double>-;
#pragma link C++ class UniformAngleParameter<double>-;
#pragma link C++ class std::vector<std::pair<double, double>>-;

#pragma link C++ class Prediction-;
#pragma link C++ class CovariatedPrediction-;
#pragma link C++ class Product-;
#pragma link C++ class Sum-;
#pragma link C++ class WeightedSum-;
#pragma link C++ class Points-;
#pragma link C++ class Histogram-;
#pragma link C++ class LinearInterpolator-;
#pragma link C++ class FillLike-;

#pragma link C++ class CovarianceToyMC-;
#pragma link C++ class PoissonToyMC-;

#pragma link C++ class Neutrino-;
#pragma link C++ class OscillationExpressions-;
#pragma link C++ class PMNSExpressions-;

#pragma link C++ class OscProbAveraged-;
#pragma link C++ class OscProb2nu-;
#pragma link C++ class OscProbPMNS-;
#pragma link C++ class OscProbPMNSDecoh-;
#pragma link C++ class EvisToEe-;
#pragma link C++ class IbdInteraction-;
#pragma link C++ class IbdZeroOrder-;
#pragma link C++ class IbdFirstOrder-;

#pragma link C++ class GaussLegendre-;
#pragma link C++ class GaussLegendreHist-;
#pragma link C++ class GaussLegendre2d-;
#pragma link C++ class GaussLegendre2dHist-;

#pragma link C++ class EnergyResolution-;

#pragma link C++ class ReactorNormAbsolute-;
#pragma link C++ class ReactorNorm-;
#pragma link C++ class ReactorGroup-;
#pragma link C++ class GaussianPeakWithBackground-;
#pragma link C++ class C14Spectrum-;
#pragma link C++ class GeoNeutrinoFluxNormed-;


#endif
