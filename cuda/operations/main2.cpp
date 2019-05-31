#include <iostream>


template<typename T> 
void check(T* var) {
	std::cout<< sizeof(var) <<std::endl;
}

int main() {

	short** v1;
	std::cout << sizeof(v1) << std::endl;
	short* v2;
	short v3;

	std::cout << sizeof(v2) << std::endl << sizeof(v3) << std::endl << "template finc:" << std::endl;

	check(v1);
	check(v2);
	return 0;
}
