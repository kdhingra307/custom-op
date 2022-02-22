/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/
#include "tensorflow/core/summary/summary_file_writer.h"

#include "absl/strings/match.h"
#include "absl/strings/str_cat.h"
#include "tensorflow/core/framework/graph.pb.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/resource_mgr.h"
#include "tensorflow/core/framework/summary.pb.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/lib/io/path.h"
#include "tensorflow/core/summary/summary_converter.h"
#include "tensorflow/core/util/events_writer.h"
#include "tensorflow/core/util/ptr_util.h"
#include "tensorflow/core/framework/register_types.h"
#include <stdexcept>
#include "./influxdb.hpp"
#include <string>

namespace tensorflow{

  namespace{
  
    template <typename T>
Status TensorValueAt(Tensor t, int64_t i, T* out) {
#define CASE(I)                            \
  case DataTypeToEnum<I>::value:           \
    *out = static_cast<T>(t.flat<I>()(i)); \
    break;
#define COMPLEX_CASE(I)                           \
  case DataTypeToEnum<I>::value:                  \
    *out = static_cast<T>(t.flat<I>()(i).real()); \
    break;
  // clang-format off
  switch (t.dtype()) {
    TF_CALL_bool(CASE)
    TF_CALL_half(CASE)
    TF_CALL_float(CASE)
    TF_CALL_double(CASE)
    TF_CALL_int8(CASE)
    TF_CALL_int16(CASE)
    TF_CALL_int32(CASE)
    TF_CALL_int64(CASE)
    TF_CALL_uint8(CASE)
    TF_CALL_uint16(CASE)
    TF_CALL_uint32(CASE)
    TF_CALL_uint64(CASE)
    TF_CALL_complex64(COMPLEX_CASE)
    TF_CALL_complex128(COMPLEX_CASE)
    default:
        return errors::Unimplemented("SummaryFileWriter ",
                                     DataTypeString(t.dtype()),
                                     " not supported.");
  }
  // clang-format on
  return Status::OK();
#undef CASE
#undef COMPLEX_CASE
}



class InfluxWriter : public SummaryWriterInterface {
 public:
  InfluxWriter(const string& experiment, int max_queue, Env* env)
      : SummaryWriterInterface(),
        is_initialized_(false),
        max_queue_(max_queue),
        experiment_(experiment),
        mode_(experiment),
        env_(env) {}

  Status Initialize(const string& url, const int port, const string& token, const string& project) {
    si = influxdb_cpp::server_info(url, port, "tkrd", token, project);
    is_initialized_ = true;
    
    return Status::OK();
  }

  Status Flush() override  TF_EXCLUSIVE_LOCKS_REQUIRED(mu_) {
    
    std::string a;
    for (const auto &piece : queue_) a += piece + "\n";
    
    std::cout << std::endl << a <<std::endl;

    //queue_.clear();

    //return InternalFlush();

    //std::cout << "flush" << std::endl;

    return Status::OK();
  }

 // Status InternalFlush() TF_EXCLUSIVE_LOCKS_REQUIRED(mu_) {
   // queue_.clear();

    //return Status::OK();
  //}


  Status WriteEvent(std::unique_ptr<Event> event) override {
    return Status::OK();
  }

  ~InfluxWriter() override {
    (void)Flush();  // Ignore errors.
  }
  Status WriteTensor(int64_t global_step, Tensor t, const string& tag,
                     const string& serialized_metadata) override { 
    
    if (serialized_metadata.empty())
      return errors::FailedPrecondition("Metadata is empty");
    
    float value;
    string type;
    SummaryMetadata metadata;
    
    metadata.ParseFromString(serialized_metadata);
    type = metadata.plugin_data().plugin_name();

    
    if (type.compare("scalars")==0){
         float value;
         TF_RETURN_IF_ERROR(TensorValueAt<float>(t, 0, &value));
   
         queue_.emplace_back(influxdb_cpp::builder()
                                .meas(experiment_)
                                .tag("metric", tag)
                                .tag("mode", "mode_")
                                .tag("type", "scalar")
                                .tag("step", std::to_string(global_step))
                                .field("value", value)
                                .timestamp(env_->NowNanos())
                                .data());
    }else if(type.compare("histograms")==0){
      influxdb_cpp::detail::field_caller* dummy;
      influxdb_cpp::builder tmp = influxdb_cpp::builder();
      dummy = (influxdb_cpp::detail::field_caller*) &tmp.meas(experiment_)
                  .tag("metric", tag)
                  .tag("mode", mode_)
                  .tag("type", "histogram")
                  .tag("step", std::to_string(global_step));
      
      for (int i=0;i<t.NumElements()/3;i++){
        double left, right, value;
        TF_RETURN_IF_ERROR(TensorValueAt<double>(t, i, &left));
        TF_RETURN_IF_ERROR(TensorValueAt<double>(t, i, &right));
        TF_RETURN_IF_ERROR(TensorValueAt<double>(t, i, &value));
        
        if (value != 0)
            dummy->field(std::to_string(left), value);
      }
      
      queue_.emplace_back(dummy->timestamp(env_->NowNanos()).data());
      tmp.clear_stream();
    
      Flush();

    }else
      return errors::FailedPrecondition("Metadata is not scalars or histograms");
      
    if (queue_.size() > max_queue_ ) {
      Flush();
    }

    return Status::OK();
  }

  Status WriteScalar(int64_t global_step, Tensor t,
                     const string& tag) override {

    return errors::FailedPrecondition("Scalars are processed using tensor function only, this function should not be called");
  }


   Status WriteHistogram(int64_t global_step, Tensor t,
                        const string& tag) override {
 
    return errors::FailedPrecondition("Histograms are processed using tensor function only, this function should not be called");
  }

  Status WriteImage(int64_t global_step, Tensor t, const string& tag,
                    int max_images, Tensor bad_color) override {
 
    return errors::FailedPrecondition("Images are not supported");
  }

  Status WriteAudio(int64_t global_step, Tensor t, const string& tag,
                    int max_outputs, float sample_rate) override {
 
    return errors::FailedPrecondition("Audio files are not supported");
  }

  Status WriteGraph(int64_t global_step,
                    std::unique_ptr<GraphDef> graph) override {
 
    return errors::FailedPrecondition("Graphs are not supported, yet");
  }


  string DebugString() const override { return "SummaryFileWriter"; }

 private:
  double GetWallTime() {
    return static_cast<double>(env_->NowMicros()) / 1.0e6;
  }

  bool is_initialized_;
  const int max_queue_;
  uint64 last_flush_;
  const string& experiment_;
  const string& mode_;
  Env* env_;
  mutex mu_;
  influxdb_cpp::server_info si;
  std::vector<string> queue_;
  //TF_GUARDED_BY(mu_);
  // A pointer to allow deferred construction.
};

}  // namespace

REGISTER_KERNEL_BUILDER(Name("InfluxWriter").Device(DEVICE_CPU),
                        ResourceHandleOp<SummaryWriterInterface>);

}
