#ifndef TENSORFLOW_CORE_SUMMARY_SUMMARY_FILE_WRITER_H_
#define TENSORFLOW_CORE_SUMMARY_SUMMARY_FILE_WRITER_H_

#include "tensorflow/core/kernels/summary_interface.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/types.h"

namespace tensorflow {

/// \brief Creates SummaryWriterInterface which writes to a file.
///
/// The file is an append-only records file of tf.Event protos. That
/// makes this summary writer suitable for file systems like GCS.
///
/// It will enqueue up to max_queue summaries, and flush at least every
/// flush_millis milliseconds. The summaries will be written to the
/// directory specified by logdir and with the filename suffixed by
/// filename_suffix. The caller owns a reference to result if the
/// returned status is ok. The Env object must not be destroyed until
/// after the returned writer.
Status CreateSummaryFileWriter(int max_queue, int flush_millis,
                               const string& logdir,
                               const string& filename_suffix, Env* env,
                               SummaryWriterInterface** result);

}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_SUMMARY_SUMMARY_FILE_WRITER_H_
