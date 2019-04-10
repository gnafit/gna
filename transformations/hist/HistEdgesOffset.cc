#include "HistEdgesOffset.hh"
#include "TypeClasses.hh"
#include "fmt/format.h"

#include <algorithm>
#include <iterator>

void HistEdgesOffset::init(){
    auto trans = this->transformation_("histedges");
    trans.input("hist", /*inactive*/true)
         .output("points")                 // 0
         .output("points_truncated")       // 1
         .output("hist_truncated")         // 2
         .types(new TypeClasses::CheckKindT<double>(DataKind::Hist), new TypeClasses::CheckNdimT<double>(1))
         .types(&HistEdgesOffset::types)
         .func(&HistEdgesOffset::func);
         ;

    if(m_threshold){
        trans.output("hist_threshold")     // 3
             .output("hist_offset")        // 4
             .output("points_threshold")   // 5
             .output("points_offset")      // 6
             ;
    }

}

void HistEdgesOffset::types(TypesFunctionArgs& fargs){
    auto& hist_input = fargs.args[0];
    auto& edges_input = hist_input.edges;

    if(!m_offset){
        auto it = std::upper_bound(edges_input.begin(), edges_input.end(), m_threshold.value());
        if( it==edges_input.end() || it==edges_input.begin() ){
            throw fargs.args.error(hist_input, "Unable to find proper offset");
        }

        m_offset = std::distance(edges_input.begin(), it-1);
    }

    size_t size_in = hist_input.shape[0];
    if(m_offset>size_in){
        throw fargs.args.error(hist_input, "Input histogram has insufficient number of bins for given offset");
    }
    size_t newsize = size_in-m_offset.value();

    auto& rets = fargs.rets;

    /// Fill the regular outputs
    auto& points           = rets[0];
    auto& points_truncated = rets[1];
    auto& hist_truncated   = rets[2];

    points.points().shape(size_in+1);
    points_truncated.points().shape(newsize+1);
    hist_truncated.hist().edges(newsize+1, edges_input.data()+m_offset.value());

    //m_dt_hist_input       = hist_input;
    //m_dt_hist_truncated   = hist_truncated;
    //m_dt_points           = points;
    //m_dt_points_truncated = points_truncated;

    if(!m_threshold){
        return;
    }

    /// Fill the offset/threshold outputs
    auto& hist_threshold   = rets[3];
    auto& hist_offset      = rets[4];
    auto& points_threshold = rets[5];
    auto& points_offset    = rets[6];

    points_threshold.points().shape(newsize+1);
    points_offset.points().shape(newsize+1);

    auto edges_threshold=hist_truncated.edges;
    if( m_threshold.value()<=edges_threshold[0] || m_threshold.value()>=edges_threshold[1] ){
        auto message = fmt::format("Threshold {} is not located "
                                   "in a first bin ({}) of the truncated histogram (excluding edges): ({}, {})",
                                   m_threshold.value(), m_offset.value(), edges_threshold[0], edges_threshold[1]);
        throw fargs.args.error(hist_input, message);
    }
    auto threshold = edges_threshold[0] = m_threshold.value();

    auto edges_offset = edges_threshold;
    std::transform(edges_offset.begin(), edges_offset.end(), edges_offset.begin(), [threshold](double v){ return v-threshold; });

    hist_threshold.hist().edges(edges_threshold);
    hist_offset.hist().edges(edges_offset);

    //m_dt_hist_threshold = hist_threshold;
}

void HistEdgesOffset::func(FunctionArgs& fargs){
    auto& edges_input = fargs.args[0].type.edges;

    auto& rets = fargs.rets;

    /// Fill the regular outputs
    auto& points           = rets[0];
    auto& points_truncated = rets[1];
    auto& hist_truncated   = rets[2];

    auto& edges_truncated = hist_truncated.type.edges;

    points.x           = Eigen::Map<const Eigen::ArrayXd>(edges_input.data(), edges_input.size());
    points_truncated.x = Eigen::Map<const Eigen::ArrayXd>(edges_truncated.data(), edges_truncated.size());
    hist_truncated.x   = 0.0;

    if(!m_threshold){
        return;
    }

    /// Fill the offset/threshold outputs
    auto& hist_threshold   = rets[3];
    auto& hist_offset      = rets[4];
    auto& points_threshold = rets[5];
    auto& points_offset    = rets[6];

    auto& edges_threshold = hist_threshold.type.edges;
    auto& edges_offset    = hist_offset.type.edges;

    points_threshold.x = Eigen::Map<const Eigen::ArrayXd>(edges_threshold.data(), edges_threshold.size());
    points_offset.x    = Eigen::Map<const Eigen::ArrayXd>(edges_offset.data(), edges_offset.size());
    hist_threshold.x   = 0.0;
    hist_offset.x      = 0.0;

    rets.untaint();
    rets.freeze();
}

//void HistEdgesOffset::add_transformation(double fillvalue){
    //auto trans = this->transformation_("view")
                 //.input("original")
                 //.input("rear")
                 //.output("result")
                 //.types(new TypeClasses::CheckNdimT<double>(1))
                 //.types(&HistEdgesOffset::viewTypes)
                 //.func(&HistEdgesOffset::viewFunc);

    //if(m_fillvalue){
        //if(m_fillvalue!=fillvalue){
            //throw std::runtime_error("Inconsistent fill value.");
        //}
    //}
    //else{
        //m_fillvalue = fillvalue;
    //}
//}

//void HistEdgesOffset::viewTypes(TypesFunctionArgs& fargs){
    //auto& args = fargs.args;
    //auto& rets = fargs.rets;
    //for (size_t i = 0; i < args.size(); ++i) {
        //auto& dt=args[dt];

        //switch(dt.kind){
            //case DataKind::Points:
                //if(dt!=m_dt_points_truncated){
                    //throw args.error(dt, "Inconsistent input datatype (Points)");
                //}
                //rets[i]=m_dt_points;
                //break;

            //case DataKind::Points:
                //if(dt!=m_dt_hist_truncated && (!m_threshold в
                                               //dt!=m_dt_hist_threshold)){
                    //throw args.error(dt, "Inconsistent input datatype (Hist)");
                //}
                //rets[i]=m_dt_hist_input;
                //break;

            //default:
                //continue;
        //}

    //}
//}

//void HistEdgesOffset::viewFunc(FunctionArgs& fargs){
    //fargs.args.touch();
//}

