#ifndef __TORCH_API_H__
#define __TORCH_API_H__
#include<stdint.h>

#ifdef __cplusplus
thread_local char *torch_last_err = nullptr;

extern "C" {
typedef torch::Tensor *tensor;
typedef torch::Scalar *scalar;
typedef torch::optim::Optimizer *optimizer;
typedef torch::jit::script::Module *module;
typedef torch::jit::IValue *ivalue;
#define PROTECT(x) \
  try { \
    x \
  } catch (const exception& e) { \
      torch_last_err = strdup(e.what()); \
  }
#else
typedef void *tensor;
typedef void *optimizer;
typedef void *scalar;
typedef void *module;
typedef void *ivalue;
#endif

char *get_and_reset_last_err(); // thread-local
void at_manual_seed(int64_t);
int at_new_tensor(tensor *);
int at_tensor_of_blob(tensor *, void *data, int64_t *dims, size_t ndims, int64_t *strides, size_t nstrides, int type, int device);
int at_tensor_of_data(tensor *, void *vs, int64_t *dims, size_t ndims, size_t element_size_in_bytes, int type);
void at_copy_data(tensor tensor, void *vs, size_t numel, size_t element_size_in_bytes);
int at_shallow_clone(tensor *, tensor);

void *at_data_ptr(tensor);
int at_defined(int *, tensor);
int at_is_mkldnn(int *, tensor);
int at_is_sparse(int *, tensor);
int at_device(int *, tensor);
int at_dim(size_t *, tensor);
void at_shape(tensor, int64_t *);
void at_stride(tensor, int64_t *);
int at_scalar_type(int *, tensor);

void at__amp_non_finite_check_and_unscale(tensor, tensor, tensor);

void at_autocast_clear_cache();
int at_autocast_decrement_nesting(int *);
int at_autocast_increment_nesting(int *);
int at_autocast_is_enabled(bool *);
int at_autocast_set_enabled(bool *, bool b);

void at_backward(tensor, int, int);
int at_requires_grad(int *, tensor);
int at_grad_set_enabled(int *, int);

int at_get(tensor *, tensor, int index);
void at_fill_double(tensor, double);
void at_fill_int64(tensor, int64_t);

int at_double_value_at_indexes(double *, tensor, int64_t *indexes, int indexes_len);
int at_int64_value_at_indexes(int64_t *, tensor, int64_t *indexes, int indexes_len);
void at_set_double_value_at_indexes(tensor, int *indexes, int indexes_len, double v);
void at_set_int64_value_at_indexes(tensor, int *indexes, int indexes_len, int64_t v);

void at_copy_(tensor dst, tensor src);

void at_print(tensor);
char *at_to_string(tensor, int line_size);
void at_save(tensor, char *filename);
int at_load(tensor *, char *filename);

void at_save_multi(tensor *tensors, char **tensor_names, int ntensors, char *filename);
/* [at_load_multi] takes as input an array of nullptr for [tensors]. */
void at_load_multi(tensor *tensors, char **tensor_names, int ntensors, char *filename);
/* [at_load_multi_] takes as input an array of allocation [tensors]. */
void at_load_multi_(tensor *tensors, char **tensor_names, int ntensors, char *filename);

void at_load_callback(char *filename, void *data, void (*f)(void *, char *, tensor));
void at_load_callback_with_device(char *filename, void *data, void (*f)(void *, char *, tensor), int device_id);

int at_get_num_interop_threads(int *);

int at_get_num_threads(int *);

void at_set_num_interop_threads(int n_threads);

void at_set_num_threads(int n_threads);

void at_set_qengine(int qengine);

void at_free(tensor);

void at_run_backward(tensor *tensors,
                      int ntensors,
                      tensor *inputs,
                      int ninputs,
                      tensor *outputs,
                      int keep_graph,
                      int create_graph);

int ato_adam(optimizer *, double learning_rate,
                   double beta1,
                   double beta2,
                   double weight_decay);
int ato_adamw(optimizer *, double learning_rate,
                   double beta1,
                   double beta2,
                   double weight_decay);
int ato_rms_prop(optimizer *, double learning_rate,
                       double alpha,
                       double eps,
                       double weight_decay,
                       double momentum,
                       int centered);
int ato_sgd(optimizer *, double learning_rate,
                  double momentum,
                  double dampening,
                  double weight_decay,
                  int nesterov);
void ato_add_parameters(optimizer, tensor, size_t group);
void ato_set_learning_rate(optimizer, double learning_rate);
void ato_set_momentum(optimizer, double momentum);
void ato_set_learning_rate_group(optimizer, size_t group, double learning_rate);
void ato_set_momentum_group(optimizer, size_t group, double momentum);
void ato_set_weight_decay(optimizer t, double weight_decay);
void ato_set_weight_decay_group(optimizer t, size_t group, double weight_decay);
void ato_zero_grad(optimizer);
void ato_step(optimizer);
void ato_free(optimizer);

int ats_int(scalar *, int64_t);
int ats_float(scalar *, double);
int ats_to_int(int64_t *, scalar);
int ats_to_float(double *, scalar);
char *ats_to_string(scalar);
void ats_free(scalar);

int atc_cuda_device_count(int *);
int atc_cuda_is_available(int *);
int atc_cudnn_is_available(int *);
int atc_user_enabled_cudnn(int *);
void atc_set_user_enabled_cudnn(int b);
void atc_set_benchmark_cudnn(int b);

int atm_load(module *, char *);
int atm_load_on_device(module *, char *, int device);
int atm_load_str(module *, char *, size_t sz);
int atm_load_str_on_device(module *, char *, size_t sz, int device);
int atm_forward(tensor *, module, tensor *tensors, int ntensors);
int atm_forward_(ivalue *, module,
                    ivalue *ivalues,
                    int nivalues);
int atm_method(tensor *, module,
                  char *method_name,
                  tensor *tensors,
                  int ntensors);
int atm_method_(ivalue *, module,
                   char *method_name,
                   ivalue *ivalues,
                   int nivalues);
void atm_eval(module);
void atm_train(module);
void atm_free(module);
void atm_to(module m, int device, int dtype, bool non_blocking);
void atm_save(module m, char*);
int atm_get_profiling_mode(int *);
void atm_set_profiling_mode(int);
void atm_named_parameters(module, void *data, void (*f)(void *, char *, tensor));

// This function has to be followed by a call to atm_end_tracing.
int atm_create_for_tracing(module *, char *modl_name, tensor *inputs, int ninputs);
void atm_end_tracing(module m, char *fn_name, tensor *outputs, int noutputs);

int ati_none(ivalue *);
int ati_tensor(ivalue *, tensor);
int ati_int(ivalue *, int64_t);
int ati_double(ivalue *, double);
int ati_bool(ivalue *, int);
int ati_string(ivalue *, char *);
int ati_tuple(ivalue *, ivalue *, int);
int ati_generic_list(ivalue *, ivalue *, int);
int ati_generic_dict(ivalue *, ivalue *, int);
int ati_int_list(ivalue *, int64_t *, int);
int ati_double_list(ivalue *, double *, int);
int ati_bool_list(ivalue *, char *, int);
int ati_string_list(ivalue *, char **, int);
int ati_tensor_list(ivalue *, tensor *, int);

int ati_to_tensor(tensor *, ivalue);
int ati_to_int(int64_t *, ivalue);
int ati_to_double(double *, ivalue);
char *ati_to_string(ivalue);
int ati_to_bool(int *, ivalue);
int ati_length(int *, ivalue);
int ati_tuple_length(int *, ivalue);
void ati_to_tuple(ivalue, ivalue *, int);
void ati_to_generic_list(ivalue, ivalue *, int);
void ati_to_generic_dict(ivalue, ivalue *, int);
void ati_to_int_list(ivalue, int64_t *, int);
void ati_to_double_list(ivalue, double *, int);
void ati_to_bool_list(ivalue, char *, int);
void ati_to_tensor_list(ivalue, tensor *, int);

int ati_tag(int *, ivalue);

void ati_free(ivalue);

#include "torch_api_generated.h"

#ifdef __cplusplus
};
#endif

#endif
