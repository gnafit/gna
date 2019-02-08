#pragma once
#include "TransformationFunctionArgs.hh"

namespace GNAUnitTest {
	//static bool typefunction(void (&func) (TransformationTypes::TypesFunctionArgs&), TransformationTypes::TypesFunctionArgs& fargs);
	void testMove();
	void testMove1();
}

//static bool GNAUnitTest::typefunction(void (&func) (TransformationTypes::TypesFunctionArgs&), TransformationTypes::TypesFunctionArgs& fargs){
	//try {
		//func(fargs);
	//}
	//catch(const TransformationTypes::SourceTypeError& e){
		//return true;
	//}
	//catch(const TransformationTypes::SinkTypeError& e){
		//return true;
	//}
	//catch(const std::runtime_error& e){
		//return true;
	//}
	//return false;
//}
