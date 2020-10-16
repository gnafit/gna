#include "unit_exceptions.hh"
#include "Exceptions.hh"
#include "TransformationErrors.hh"

void GNAUnitTest::test_keyerror_exception(){
	throw KeyError("somekey", "of someobject");
}

void GNAUnitTest::test_typeerror_exception(){
	throw TransformationTypes::TypeError("some message");
}
