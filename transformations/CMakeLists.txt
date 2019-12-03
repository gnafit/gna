set(TRANSFORMATION_SOURCES
  # Base
  transformations/base/FillLike.cc
  transformations/base/Histogram.cc
  transformations/base/Histogram2d.cc
  transformations/base/Points.cc
  transformations/base/View.cc
  transformations/base/ViewRear.cc
  transformations/base/ViewHistBased.cc
  transformations/base/Concat.cc
  transformations/base/Product.cc
  transformations/base/ProductBC.cc
  transformations/base/Snapshot.cc
  # Sums
  transformations/sum/MultiSum.cc
  transformations/sum/PartialSum.cc
  transformations/sum/Sum.cc
  transformations/sum/SumBroadcast.cc
  transformations/sum/WeightedSum.cc
  transformations/sum/WeightedSumP.cc
  # Ratios
  transformations/ratio/NormalizedConvolution.cc
  transformations/ratio/Convolution.cc
  transformations/ratio/PolyRatio.cc
  transformations/ratio/Ratio.cc
  # Calculus
  transformations/calculus/Derivative.cc
  transformations/calculus/Jacobian.cc
  # Detector effects
  transformations/detector/FakeGSLFunction.cc
  transformations/detector/C14Spectrum.cc
  transformations/detector/EnergyResolution.cc
  transformations/detector/HistNonlinearity.cc
  transformations/detector/HistSmear.cc
  transformations/detector/HistSmearSparse.cc
  transformations/detector/Cherenkov_Borexino.cc
  # Functions
  transformations/functions/SelfPower.cc
  transformations/functions/Exp.cc
  transformations/functions/Sin.cc
  transformations/functions/Cos.cc
  transformations/functions/Inverse.cc
  # Histograms
  transformations/hist/Bins.cc
  transformations/hist/HistEdges.cc
  transformations/hist/HistEdgesOffset.cc
  transformations/hist/HistEdgesLinear.cc
  transformations/base/PointsToHist.cc
  transformations/hist/Rebin.cc
  transformations/hist/RebinN.cc
  # Integration
  transformations/integrator/GSLSamplerGL.cc
  transformations/integrator/IntegratorBase.cc
  transformations/integrator/Integrator21Base.cc
  transformations/integrator/Integrator2Base.cc
  transformations/integrator/IntegratorGL.cc
  transformations/integrator/Integrator21GL.cc
  transformations/integrator/Integrator2GL.cc
  transformations/integrator/IntegratorRect.cc
  transformations/integrator/IntegratorTrap.cc
  # Interpolation
  transformations/interpolation/InSegment.cc
  transformations/interpolation/SegmentWise.cc
  transformations/interpolation/InterpBase.cc
  transformations/interpolation/InterpLinear.cc
  transformations/interpolation/InterpExpo.cc
  transformations/interpolation/InterpLog.cc
  transformations/interpolation/InterpLogx.cc
  transformations/interpolation/FixedPointScale.cc
  # Linear algebra
  transformations/linalg/Cholesky.cc
  transformations/linalg/Normalize.cc
  transformations/linalg/RenormalizeDiag.cc
  transformations/linalg/MatrixProduct.cc
  transformations/linalg/Transpose.cc
  # Neutrino physics
  transformations/neutrino/EvisToEe.cc
  transformations/neutrino/IbdFirstOrder.cc
  transformations/neutrino/IbdInteraction.cc
  transformations/neutrino/IbdZeroOrder.cc
  transformations/neutrino/Neutrino.cc
  transformations/neutrino/OscillationVariables.cc
  transformations/neutrino/OscProb2nu.cc
  transformations/neutrino/OscProbMatterUniform.cc
  transformations/neutrino/OscProbPMNS.cc
  transformations/neutrino/OscProb3.cc
  transformations/neutrino/OscProbPMNSDecoh.cc
  transformations/neutrino/PDGVariables.cc
  transformations/neutrino/PMNSVariables.cc
  transformations/neutrino/OscProbPMNSVariables.cc
  transformations/neutrino/ReactorGroup.cc
  transformations/neutrino/ReactorNorm.cc
  transformations/neutrino/Units.cc
  # Statistics
  transformations/stats/Chi2.cc
  transformations/stats/CovarianceToyMC.cc
  transformations/stats/CovariatedPrediction.cc
  transformations/stats/CovariatedPredictionV.cc
  transformations/stats/Covmat.cc
  transformations/stats/NormalToyMC.cc
  transformations/stats/Poisson.cc
  transformations/stats/PoissonToyMC.cc
  transformations/stats/ParCovMatrix.cc
  # Variables manipulation
  transformations/var/VarArray.cc
  transformations/var/VarArrayPreallocated.cc
  transformations/var/VarDiff.cc
  transformations/var/VarProduct.cc
  transformations/var/VarSum.cc
  transformations/var/ArraySum.cc
  #
  transformations/backgrounds/GeoNeutrinoFluxNormed.cc
  # Debug transformations
  transformations/debug/Identity.cc
  transformations/debug/DebugTransformation.cc
  transformations/debug/Dummy.cc
  transformations/debug/DummyType.cc
  transformations/debug/TrialMultiFunc.cc
  # Legacy transformations
  transformations/legacy/EnergyResolutionC.cc
  transformations/legacy/GaussLegendre.cc
  transformations/legacy/GaussLegendre2d.cc
  transformations/legacy/InterpExpoSorted.cc
  transformations/legacy/LinearInterpolator.cc
  PARENT_SCOPE
)

set(TRANSFORMATION_HEADERS
  # Base
  Bins.hh
  Concat.hh
  FillLike.hh
  Histogram.hh
  Histogram2d.hh
  Points.hh
  View.hh
  ViewRear.hh
  ViewHistBased.hh
  Product.hh
  ProductBC.hh
  Snapshot.hh
  # Sums
  PartialSum.hh
  MultiSum.hh
  Sum.hh
  SumBroadcast.hh
  WeightedSum.hh
  WeightedSumP.hh
  # Ratios
  Convolution.hh
  NormalizedConvolution.hh
  PolyRatio.hh
  Ratio.hh
  # Calculus
  Derivative.hh
  Jacobian.hh
  # Detector effects
  CovariatedPrediction.hh
  CovariatedPredictionV.hh
  HistEdges.hh
  HistEdgesOffset.hh
  HistEdgesLinear.hh
  InSegment.hh
  SegmentWise.hh
  InterpExpo.hh
  InterpLinear.hh
  InterpLog.hh
  InterpLogx.hh
  FixedPointScale.hh
  Covmat.hh
  Cholesky.hh
  Chi2.hh
  Poisson.hh
  Normalize.hh
  CovarianceToyMC.hh
  NormalToyMC.hh
  PoissonToyMC.hh
  # Functions
  Inverse.hh
  Exp.hh
  Sin.hh
  Cos.hh
  SelfPower.hh
  # Histogram
  Rebin.hh
  RebinN.hh
  PointsToHist.hh
  VarArray.hh
  VarArrayPreallocated.hh
  VarProduct.hh
  VarSum.hh
  VarDiff.hh
  ArraySum.hh
  Minimizable.hh
  Neutrino.hh
  Units.hh
  OscillationVariables.hh
  PMNSVariables.hh
  OscProbPMNSVariables.hh
  OscProb2nu.hh
  OscProbPMNS.hh
  OscProb3.hh
  OscProbPMNSDecoh.hh
  OscProbMatterUniform.hh
  EvisToEe.hh
  IbdInteraction.hh
  IbdZeroOrder.hh
  IbdFirstOrder.hh
  ReactorNorm.hh
  ReactorGroup.hh
  IntegratorBase.hh
  Integrator21Base.hh
  Integrator2Base.hh
  IntegratorRect.hh
  IntegratorTrap.hh
  GSLSamplerGL.hh
  GaussLegendre.hh
  GaussLegendre2d.hh
  EnergyResolution.hh
  EnergyResolutionC.hh
  HistNonlinearity.hh
  Cherenkov_Borexino.hh
  HistSmear.hh
  HistSmearSparse.hh
  RenormalizeDiag.hh
  C14Spectrum.hh
  ParCovMatrix.hh
  MatrixProduct.hh
  Transpose.hh
  GeoNeutrinoFluxNormed.hh
  # Debug
  DebugTransformation.hh
  Dummy.hh
  DummyType.hh
  Identity.hh
  TrialMultiFunc.hh
  # Legacy
  IntegratorGL.hh
  Integrator21GL.hh
  Integrator2GL.hh
  InterpExpoSorted.hh
  LinearInterpolator.hh
  InterpStrategy.hh
  InterpBase.hh
  PARENT_SCOPE
)
