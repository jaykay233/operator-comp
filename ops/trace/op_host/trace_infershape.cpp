/*!
 * \file trace_infershape.cpp
 * \brief Trace 算子形状推导实现
 *
 * Trace 输出为标量（用 shape (1,) 承载），与输入 shape 无关。
 */

#include "register/op_impl_registry.h"
#include "exe_graph/runtime/infer_shape_context.h"

using namespace ge;

namespace ops {

static ge::graphStatus InferShapeTrace(gert::InferShapeContext* context)
{
    gert::Shape* output_shape = context->GetOutputShape(0);
    if (output_shape == nullptr) {
        return ge::GRAPH_FAILED;
    }
    output_shape->SetDimNum(1);
    output_shape->SetDim(0, 1);
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(Trace).InferShape(InferShapeTrace);

} // namespace ops
