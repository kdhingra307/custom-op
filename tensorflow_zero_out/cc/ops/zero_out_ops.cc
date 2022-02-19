#include "tensorflow/core/framework/common_shape_fns.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference.h"

namespace tensorflow {

REGISTER_OP("InfluxWriter")
    .Output("writer: resource")
    .Attr("shared_name: string = ''")
    .Attr("container: string = ''")
    .SetShapeFn(shape_inference::ScalarShape);

REGISTER_OP("CreateTesttFileWriter")
    .Input("writer: resource")
    .Input("url: string")
    .Input("port: int32")
    .Input("project: string")
    .Input("token: string")
    .Input("experiment: string")
    .Input("max_queue: int32")
    .SetShapeFn(shape_inference::NoOutputs);
}
