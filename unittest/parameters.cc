#include "parameters.hh"
#include "taintflag.hh"

bool GNAUnitTest::freeze(changeable* obj){
	try {
		obj->freeze();
	}
	catch(const std::runtime_error& e){
		return true;
	}
	return false;
}

