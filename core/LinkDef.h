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

#pragma link C++ class PredictionSet+;

#endif
