#pragma once

#include <vector>
#include <optional>
#include "GNAObject.hh"

/**
 * @brief Transformation object holding a static 1- or 2-dimensional array.
 *
 * Outputs:
 *   - `points.points` - 1- or 2- dimensional array with fixed data.
 *
 * @author Maxim Gonchar
 * @date 2015
 */
class View: public GNAObject,
            public TransformationBind<View> {
private:
public:
    View(size_t start=0u);
    View(size_t start, size_t len);
    View(SingleOutput* output, size_t start=0u);
    View(SingleOutput* output, size_t start, size_t len);

protected:
    void types(TypesFunctionArgs& fargs);
    void init();

    size_t m_start=0u;
    std::optional<size_t> m_len;
};

