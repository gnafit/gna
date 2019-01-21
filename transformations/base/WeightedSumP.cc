#include "WeightedSumP.hh"
#include "TypesFunctions.hh"

WeightedSumP::WeightedSumP(const OutputDescriptor::OutputDescriptors& outputs) : WeightedSumP()
{
  for (size_t i{0}, j{1}; j<outputs.size();) {
    add(*outputs[i], *outputs[j]);
    i+=2; j+=2;
  }
}

WeightedSumP::WeightedSumP() {
  auto sum = transformation_("sum")
    .output("sum")
    .label("wsum")
    .types(&WeightedSumP::check)
    .func(&WeightedSumP::sum);
}

void WeightedSumP::check(TypesFunctionArgs& fargs){
  auto& args=fargs.args;
  auto remainder = args.size()%2;
  if(remainder){
    return;
  }

  DataType dtsingle = DataType().points().shape(1);
  auto dt=args[1];
  for (size_t i{0}, j{1}; j<args.size();) {
    auto& weight = args[i];
    auto& array  = args[j];

    if(weight!=dtsingle){
      printf("Current data type: ");
      weight.dump();
      throw args.error(weight, fmt::format("Arg {0} (weight) should be points of dimension 1", i));
    }

    if(array!=dt){
      printf("First data type: ");
      dt.dump();
      printf("Current data type: ");
      array.dump();
      throw args.error(array, fmt::format("Arg {0} (array) should be consistent with others", j));
    }

    i+=2; j+=2;
  }

  fargs.rets[0]=dt;
}

void WeightedSumP::sum(FunctionArgs& fargs){
  auto& args=fargs.args;
  auto& ret=fargs.rets[0].x;

  auto& weight0 = args[0].x(0);
  auto& array0  = args[1].x;
  ret=weight0*array0;
  if(args.size()>2){
    for (size_t i{2}, j{3}; j<args.size();) {
      auto& weight = args[i].x(0);
      auto& array  = args[j].x;
      ret+=weight*array;
      i+=2; j+=2;
    }
  }
}

void WeightedSumP::add(SingleOutput &a, SingleOutput &b) {
  add(a);
  add(b);
}

void WeightedSumP::add(SingleOutput &out) {
  auto trans=transformations.back();
  auto ninputs=trans.inputs.size();
  trans.input(fmt::format("input_{0:02d}", ninputs), out);
}
