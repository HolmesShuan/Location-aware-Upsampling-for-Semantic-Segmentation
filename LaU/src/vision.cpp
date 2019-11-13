
#include "location_aware_upsampling.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("location_aware_upsampling_forward", &location_aware_upsampling_forward, "location_aware_upsampling_forward");
  m.def("location_aware_upsampling_backward", &location_aware_upsampling_backward, "location_aware_upsampling_backward");
  m.def("location_determined_upsampling_forward", &location_determined_upsampling_forward, "location_determined_upsampling_forward");
  m.def("location_determined_upsampling_backward", &location_determined_upsampling_backward, "location_determined_upsampling_backward");
  m.def("location_determined_upsampling_multi_output_forward", &location_determined_upsampling_multi_output_forward, "location_determined_upsampling_multi_output_forward");
  m.def("location_determined_upsampling_multi_output_backward", &location_determined_upsampling_multi_output_backward, "location_determined_upsampling_multi_output_backward");
}
