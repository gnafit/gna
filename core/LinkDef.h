#ifdef __CINT__

#pragma link off all globals;
#pragma link off all classes;
#pragma link off all functions;

//
// A collection of classes
//

#pragma link C++ class TransformationDescriptor+;
#pragma link C++ class InputDescriptor+;
#pragma link C++ class OutputDescriptor+;
#pragma link C++ class TransformationDescriptor::Inputs+;
#pragma link C++ class TransformationDescriptor::Outputs+;

#pragma link C++ class VariableDescriptor+;
#pragma link C++ class EvaluableDescriptor+;
#pragma link C++ class EvaluableDescriptor::Sources+;

#pragma link C++ class GNAObject;
#pragma link C++ class GNAObject::Variables+;
#pragma link C++ class GNAObject::Evaluables+;

#pragma link C++ class ExpressionsProvider+;

#pragma link C++ class ParameterWrapper+;
#pragma link C++ class Uncertain+;
#pragma link C++ class GaussianValue+;
#pragma link C++ class Parameter+;
#pragma link C++ class GaussianParameter+;
#pragma link C++ class UniformAngleParameter+;
#pragma link C++ typedef LimitsPair;
#pragma link C++ class std::vector<LimitsPair>+;

#pragma link C++ class PredictionSet+;
#pragma link C++ class Product+;
#pragma link C++ class PointSet+;

#pragma link C++ class OscillationExpressions+;

#pragma link C++ class OscProb2nu+;
#pragma link C++ class IbdInteraction+;
#pragma link C++ class IbdZeroOrder+;
#pragma link C++ class IbdFirstOrder+;

#pragma link C++ class GaussLegendre+;
#pragma link C++ class GaussLegendre2d+;

#endif
