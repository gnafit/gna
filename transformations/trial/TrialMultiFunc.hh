#pragma once

#include <iostream>
#include <vector>
#include "GNAObject.hh"
#include "TypesFunctions.hh"

using std::cout;
using std::endl;

/**
 * @brief TrialMultiFunc transformation for testing switchable functions
 *
 * @author Maxim Gonchar
 * @date 18.07.2018
 */
class TrialMultiFunc: public GNASingleObject,
                      public TransformationBind<TrialMultiFunc> {
public:
  TrialMultiFunc(){
    transformation_("multifunc")
      // Any input
      .input("inp")
      // Output of the same shape as input
      .output("out")
      // TypesFunction to define output shape
      .types(TypesFunctions::passAll)
      // TypesFunction to define storage, common for all implementations
      .types([](TypesFunctionArgs& fargs){
            fargs.ints[0] = DataType().points().shape(1);
            printf("Initialize common storage ([0]: 1x1 points): %i\n", (int)fargs.ints.size());
             })
      // MemTypesFunction to define another storage, common for all implementations
      .types(&TrialMultiFunc::memTypesFunction)
      // Main Function implementation:
      .func([](FunctionArgs& fargs){
            printf("Call main Function:");
            printf(
                   "\n  - num=1"
                   "\n  - write num to ints[0]"
                   "\n  - write num*num to ints[1]"
                   "\n  - write num*num to ints[1]"
                   "\n  - write args[0]*ints[0] to rets[0]\n"
                  );
            fargs.ints[0].x = 1;
            fargs.ints[1].x = fargs.ints[0].x(0)*fargs.ints[0].x(0);
            fargs.rets[0].x = fargs.args[0].x*fargs.ints[0].x(0);
            printf("  input[0]:\n");
            cout<<fargs.args[0].x<<endl;
            printf("  output[0]:\n");
            cout<<fargs.rets[0].x<<endl;
            printf("  internal[0]:\n");
            cout<<fargs.ints[0].x<<endl;
            printf("  internal[1]:\n");
            cout<<fargs.ints[1].mat<<endl;
            })
      // Main Function's StorageTypesFunction
      .storage([](StorageTypesFunctionArgs& fargs){
             printf("Initialize main storage (nothing): %i\n", (int)fargs.ints.size());
             })
      // Secondary Function implementation
      .func("secondary", [](FunctionArgs& fargs){
            printf("Call secondary Function:");
            printf(
                   "\n  - num=2"
                   "\n  - write num to ints[0]"
                   "\n  - write num*num to ints[1]"
                   "\n  - write num*num to ints[1]"
                   "\n  - write args[0]*ints[0] to rets[0]"
                   "\n  - write args[0] to ints[2]\n"
                  );
            fargs.ints[0].x = 2;
            fargs.ints[1].x = fargs.ints[0].x(0)*fargs.ints[0].x(0);
            fargs.rets[0].x = fargs.args[0].x*fargs.ints[0].x(0);
            fargs.ints[2].x = fargs.args[0].x;
            printf("  input[0]:\n");
            cout<<fargs.args[0].x<<endl;
            printf("  output[0]:\n");
            cout<<fargs.rets[0].x<<endl;
            printf("  internal[0]:\n");
            cout<<fargs.ints[0].x<<endl;
            printf("  internal[1]:\n");
            cout<<fargs.ints[1].mat<<endl;
            printf("  internal[2]:\n");
            cout<<fargs.ints[2].x<<endl;
      })
      // Secondary Function StorageTypesFunction
      .storage("secondary", [](StorageTypesFunctionArgs& fargs){
            fargs.ints[2] = fargs.args[0];
            printf("Initialize secondary storage (clone arg[0] to ints[2]): %i\n", (int)fargs.ints.size());
            })
      // Secondary MemFunction implementation
      .func("secondaryMem", &TrialMultiFunc::memFunction)
      // Secondary MemStorageTypesFunction
      .storage("secondaryMem", &TrialMultiFunc::memStorageTypesFunction)
      // Third party Function implementation
      .func("thirdparty", [](FunctionArgs& fargs){
            printf("Call third party Function:");
            printf(
                   "\n  - num=4"
                   "\n  - write num to ints[0]"
                   "\n  - write num*num to ints[1]"
                   "\n  - write num*num to ints[1]"
                   "\n  - write args[0]*ints[0] to rets[0]"
                   "\n  - write args[0] to ints[2]\n"
                  );
            fargs.ints[0].x = 4;
            fargs.ints[1].x = fargs.ints[0].x(0)*fargs.ints[0].x(0);
            fargs.rets[0].x = fargs.args[0].x*fargs.ints[0].x(0);
            fargs.ints[2].x = fargs.args[0].x;
            printf("  input[0]:\n");
            cout<<fargs.args[0].x<<endl;
            printf("  output[0]:\n");
            cout<<fargs.rets[0].x<<endl;
            printf("  internal[0]:\n");
            cout<<fargs.ints[0].x<<endl;
            printf("  internal[1]:\n");
            cout<<fargs.ints[1].mat<<endl;
            printf("  internal[2]:\n");
            cout<<fargs.ints[2].x<<endl;
      })
      // Third party Function StorageTypesFunction
      .storage("thirdparty", [](StorageTypesFunctionArgs& fargs){
               fargs.ints[2] = fargs.args[0];
               printf("Initialize third party storage (clone arg[0] to ints[2]): %i\n", (int)fargs.ints.size());
            });
  };

  void memFunction(FunctionArgs& fargs){
    printf("Call secondary MemFunction:");
    printf(
           "\n  - num=3"
           "\n  - write num to ints[0]"
           "\n  - write num*num to ints[1]"
           "\n  - write num*num to ints[1]"
           "\n  - write args[0]*ints[0] to rets[0]"
           "\n  - write args[0] to ints[2]\n"
          );
    fargs.ints[0].x = 3;
    fargs.ints[1].x = fargs.ints[0].x(0)*fargs.ints[0].x(0);
    fargs.rets[0].x = fargs.args[0].x*fargs.ints[0].x(0);
    fargs.ints[2].x = fargs.args[0].x;
    printf("  input[0]:\n");
    cout<<fargs.args[0].x<<endl;
    printf("  output[0]:\n");
    cout<<fargs.rets[0].x<<endl;
    printf("  internal[0]:\n");
    cout<<fargs.ints[0].x<<endl;
    printf("  internal[1]:\n");
    cout<<fargs.ints[1].mat<<endl;
    printf("  internal[2]:\n");
    cout<<fargs.ints[2].x<<endl;
  }

  void memTypesFunction(TypesFunctionArgs& fargs){
    fargs.ints[1] = DataType().points().shape(2,2);
    printf("Initialize common storage (mem, [1]: 2x2 points): %i\n", (int)fargs.ints.size());
  }

  void memStorageTypesFunction(StorageTypesFunctionArgs& fargs){
      fargs.ints[2] = fargs.args[0];
      printf("Initialize secondary storage (mem, clone arg[0] to ints[2]): %i\n", (int)fargs.ints.size());
  }
};
