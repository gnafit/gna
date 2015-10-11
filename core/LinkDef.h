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
#pragma link C++ class EvaluableDescriptor::Sources+;

#pragma link C++ class GNAObject;
#pragma link C++ class GNAObject::Variables+;
#pragma link C++ class GNAObject::Evaluables+;
#pragma link C++ class GNAObject::Transformations+;
#pragma link C++ class GNASingleObject;

#pragma link C++ class ExpressionsProvider+;

#pragma link C++ class ParameterWrapper<double>+;
#pragma link C++ class Uncertain<double>+;
#pragma link C++ class Uncertain<std::complex<double>>+;
#pragma link C++ class GaussianValue<double>+;
#pragma link C++ class GaussianValue<std::complex<double>>+;
#pragma link C++ class Parameter<double>+;
#pragma link C++ class GaussianParameter<double>+;
#pragma link C++ class UniformAngleParameter<double>+;
#pragma link C++ class std::vector<std::pair<double, double>>+;

#pragma link C++ class PredictionSet+;
#pragma link C++ class Product+;
#pragma link C++ class WeightedSum+;
#pragma link C++ class PointSet+;
#pragma link C++ class Histogram+;
#pragma link C++ class LinearInterpolator+;

#pragma link C++ class Neutrino+;
#pragma link C++ class OscillationExpressions+;
#pragma link C++ class PMNSExpressions+;

#pragma link C++ class OscProb2nu+;
#pragma link C++ class OscProbPMNS+;
#pragma link C++ class IbdInteraction+;
#pragma link C++ class IbdZeroOrder+;
#pragma link C++ class IbdFirstOrder+;

#pragma link C++ class GaussLegendre+;
#pragma link C++ class GaussLegendre2d+;

#pragma link C++ class EnergyResolution+;

#endif
