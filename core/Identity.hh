#ifndef IDENTITY_H
#define IDENTITY_H 1

//
// Identity transformation
//
class Identity: public GNASingleObject,
                public Transformation<Identity> {
public:
  Identity(){
    transformation_(this, "identity")
      .input("source")
      .output("target")
      .types(Atypes::pass<0,0>)
      .func([](Args args, Rets rets){ rets[0].x = args[0].x; })
      ;
  };
};

#endif
