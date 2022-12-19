#include<torch/csrc/autograd/engine.h>
#include<torch/csrc/jit/runtime/graph_executor.h>
#include <torch/csrc/jit/passes/fixup_trace_scope_blocks.h>
#include <torch/csrc/jit/passes/normalize_ops.h>
#include<torch/torch.h>
#include<ATen/autocast_mode.h>
#include<torch/script.h>
#include<stdexcept>
#include<vector>
#include "torch_api.h"

using namespace std;

char *get_and_reset_last_err() {
    char *tmp = torch_last_err;
    torch_last_err = nullptr;
    return tmp;
}

int at_manual_seed(int64_t seed) {
  PROTECT(
    torch::manual_seed(seed);
    return 0;
  )
  return 1;
}

vector<torch::Tensor> of_carray_tensor(torch::Tensor **vs, int len) {
  vector<torch::Tensor> result;
  for (int i = 0; i < len; ++i) result.push_back(*(vs[i]));
  return result;
}

c10::List<c10::optional<torch::Tensor>> of_carray_tensor_opt(torch::Tensor **vs, int len) {
  vector<c10::optional<torch::Tensor>> result;
  for (int i = 0; i < len; ++i) {
    result.push_back(vs[i] != nullptr ? c10::optional<torch::Tensor>(*(vs[i])) : c10::nullopt);
  }
  return c10::List<c10::optional<torch::Tensor>>(result);
}

at::Device device_of_int(int d) {
    if (d < 0) return at::Device(at::kCPU);
    return at::Device(at::kCUDA, /*index=*/d);
}

int at_new_tensor(tensor *out__) {
  PROTECT(
    out__[0] = new torch::Tensor();
    return 0;
  )
  return 1;
}

int at_tensor_of_blob(tensor *out__, void *data, int64_t *dims, size_t ndims, int64_t *strides, size_t nstrides, int type, int device) {
  PROTECT(
    at::TensorOptions blobOptions = at::TensorOptions().device(device_of_int(device)).dtype(torch::ScalarType(type));
    out__[0] = new torch::Tensor(torch::from_blob(data, torch::IntArrayRef(dims, ndims), torch::IntArrayRef(strides, nstrides), blobOptions));
    return 0;
  )

  return 1;
}

int at_tensor_of_data(tensor *out__, void *vs, int64_t *dims, size_t ndims, size_t element_size_in_bytes, int type) {
  PROTECT(
    torch::Tensor tensor = torch::zeros(torch::IntArrayRef(dims, ndims), torch::ScalarType(type));
    if ((int64_t)element_size_in_bytes != tensor.element_size())
      throw std::invalid_argument("incoherent element sizes in bytes");
    void *tensor_data = tensor.data_ptr();
    memcpy(tensor_data, vs, tensor.numel() * element_size_in_bytes);
    out__[0] = new torch::Tensor(tensor);
    return 0;
  )
  return 1;
}

int at_copy_data(tensor tensor, void *vs, size_t numel, size_t elt_size_in_bytes) {
  PROTECT(
    if ((int64_t)elt_size_in_bytes != tensor->element_size())
      throw std::invalid_argument("incoherent element sizes in bytes");
    if ((int64_t)numel > tensor->numel())
      throw std::invalid_argument("target numel is larger than tensor numel");
    if (tensor->device().type() != at::kCPU) {
      torch::Tensor tmp_tensor = tensor->to(at::kCPU).contiguous();
      void *tensor_data = tmp_tensor.data_ptr();
      memcpy(vs, tensor_data, numel * elt_size_in_bytes);
    }
    else {
      auto tmp_tensor = tensor->contiguous();
      void *tensor_data = tmp_tensor.data_ptr();
      memcpy(vs, tensor_data, numel * elt_size_in_bytes);
    }
    return 0;
  )
  return 1;
}

int at_shallow_clone(tensor *out__, tensor t) {
  PROTECT(
    out__[0] = new torch::Tensor(*t);
    return 0;
  )
  return 1;
}

int at_data_ptr(void **out__, tensor t) {
  PROTECT(
    out__[0] = t->data_ptr();
    return 0;
  )
  return 1;
}


int at_defined(int *out__, tensor t) {
  PROTECT(
    out__[0] = t->defined();
    return 0;
  )
  return 1;
}

int at_is_mkldnn(int *out__, tensor t) {
  PROTECT(
    out__[0] = t->is_mkldnn();
    return 0;
  )
  return 1;
}

int at_is_sparse(int *out__, tensor t) {
  PROTECT(
    out__[0] = t->is_sparse();
    return 0;
  )
  return 1;
}

int at_dim(size_t *out__, tensor t) {
  PROTECT(
    out__[0] = t->dim();
    return 0;
  )
  return 1;
}

int at_shape(tensor t, int64_t *dims) {
  PROTECT(
    int i = 0;
    for (int64_t dim : t->sizes()) dims[i++] = dim;
    return 0;
  )
  return 1;
}

int at_stride(tensor t, int64_t *dims) {
  PROTECT(
    int i = 0;
    for (int64_t dim: t->strides()) dims[i++] = dim;
    return 0;
  )
  return 1;
}

int at_scalar_type(int *out__, tensor t) {
  PROTECT(
    out__[0] = static_cast<int>(t->scalar_type());
    return 0;
  )
  return 1;
}

int at__amp_non_finite_check_and_unscale(tensor t, tensor found_inf, tensor inf_scale) {
  PROTECT(
    at::_amp_foreach_non_finite_check_and_unscale_(*t, *found_inf, *inf_scale);
    return 0;
  )
  return 1;
}

int at_autocast_clear_cache() {
  PROTECT(
    at::autocast::clear_cache();
    return 0;
  )
  return 1;
}

int at_autocast_decrement_nesting(int *out__) {
  PROTECT(
    out__[0] = at::autocast::decrement_nesting();
    return 0;
  )
  return 1;
}

int at_autocast_increment_nesting(int *out__) {
  PROTECT(
    out__[0] = at::autocast::increment_nesting();
    return 0;
  )
  return 1;
}

int at_autocast_is_enabled(bool *out__) {
  PROTECT(
    return at::autocast::is_enabled();
  )
  return -1;
}

int at_autocast_set_enabled(bool *out__, bool b) {
  PROTECT(
    bool is_enabled = at::autocast::is_enabled();
    at::autocast::set_enabled(b);
    out__[0] = is_enabled;
    return 0;
  )
  return 1;
}

int at_device(int *out__, tensor t) {
  PROTECT(
    auto device = t->device();
    if (device.type() == at::kCPU) {
      out__[0] = -1;
      return 0;
    }
    if (device.type() == at::kCUDA) {
      out__[0] = device.index();
      return 0;
    }
  )
  return 1;
}

int at_backward(tensor t, int keep_graph, int create_graph) {
  PROTECT(
    t->backward({}, keep_graph, create_graph);
    return 0;
  )
  return 1;
}

int at_requires_grad(int *out__, tensor t) {
  PROTECT(
    out__[0] = t->requires_grad();
    return 0;
  )
  return 1;
}

int at_grad_set_enabled(int *out__, int b) {
  PROTECT(
    bool is_enabled = torch::autograd::GradMode::is_enabled();
    torch::autograd::GradMode::set_enabled(b);
    out__[0] = is_enabled;
    return 0;
  )
  return 1;
}

int at_get(tensor *out__, tensor t, int index) {
  PROTECT(
    out__[0] = new torch::Tensor((*t)[index]);
    return 0;
  )
  return 1;
}

template<typename T>
int at_value_at_indexes(T *out__, tensor t, int64_t *indexes, int indexes_len) {
  PROTECT(
    torch::Tensor tensor = *t;
    for (int i = 0; i < indexes_len; ++i) {
      tensor = tensor[indexes[i]];
    }
    out__[0] = tensor.item<T>();
    return 0;
  )
  out__[0] = T();
  return 1;
}

int at_double_value_at_indexes(double *out__, tensor t, int64_t *indexes, int indexes_len) {
  return at_value_at_indexes<double>(out__, t, indexes, indexes_len);
}

int at_int64_value_at_indexes(int64_t *out__, tensor t, int64_t *indexes, int indexes_len) {
  return at_value_at_indexes<int64_t>(out__, t, indexes, indexes_len);
}

template<typename T>
int at_set_value_at_indexes(tensor t, int *indexes, int indexes_len, T v) {
  PROTECT(
    torch::Tensor tensor = *t;
    for (int i = 0; i < indexes_len; ++i) {
      tensor = tensor[indexes[i]];
    }
    tensor.fill_(v);
    return 0;
  )
  return 1;
}

int at_set_double_value_at_indexes(tensor t, int *indexes, int indexes_len, double v) {
  return at_set_value_at_indexes<double>(t, indexes, indexes_len, v);
}

int at_set_int64_value_at_indexes(tensor t, int *indexes, int indexes_len, int64_t v) {
  return at_set_value_at_indexes<int64_t>(t, indexes, indexes_len, v);
}

int at_fill_double(tensor t, double v) {
  PROTECT(
    t->fill_(v);
    return 0;
  )
  return 1;
}

int at_fill_int64(tensor t, int64_t v) {
  PROTECT(
    t->fill_(v);
    return 0;
  )
  return 1;
}

int at_print(tensor t) {
  PROTECT(
    torch::Tensor *tensor = (torch::Tensor*)t;
    cout << *tensor << endl;
    return 0;
  )
  return 1;
}

int at_to_string(char **out__, tensor t, int line_size) {
  PROTECT(
    std::ostringstream oss;
    torch::print(oss, *t, line_size);
    out__[0] = strdup(oss.str().c_str());
    return 0;
  )
  return 1;
}

int at_copy_(tensor dst, tensor src) {
  PROTECT(
    dst->copy_(*src);
    return 0;
  )
  return 1;
}

int at_save(tensor t, char *filename) {
  PROTECT(
    torch::save(*t, filename);
    return 0;
  )
  return 1;
}

int at_save_multi(tensor *tensors, char **tensor_names, int ntensors, char *filename) {
  PROTECT(
    torch::serialize::OutputArchive archive;
    for (int i = 0; i < ntensors; ++i)
      archive.write(std::string(tensor_names[i]), *(tensors[i]), /* buffer=*/ false);
    archive.save_to(filename);
    return 0;
  )
  return 1;
}

int at_load_multi(tensor *tensors, char **tensor_names, int ntensors, char *filename) {
  PROTECT(
    torch::serialize::InputArchive archive;
    archive.load_from(std::string(filename));
    vector<torch::Tensor> ts(ntensors);
    for (int i = 0; i < ntensors; ++i)
      archive.read(std::string(tensor_names[i]), ts[i]);
    // Only allocate the new tensor now so that if there is an exception raised during
    // [read], no memory has to be freed.
    for (int i = 0; i < ntensors; ++i)
      tensors[i] = new torch::Tensor(ts[i]);
    return 0;
  )
  return 1;
}

int at_load_callback(char *filename, void *data, void (*f)(void *, char *, tensor)) {
  PROTECT(
    auto module = torch::jit::load(filename);
    for (const auto &p : module.named_parameters()) {
      auto v = p.value;
      f(data, (char*)p.name.c_str(), new torch::Tensor(v));
    }
    return 0;
  )
  return 1;
}

int at_load_callback_with_device(char *filename, void *data, void (*f)(void *, char *, tensor), int device_id) {
  PROTECT(
    auto module = torch::jit::load(filename, device_of_int(device_id));
    for (const auto &p : module.named_parameters()) {
      auto v = p.value;
      f(data, (char*)p.name.c_str(), new torch::Tensor(v));
    }
    return 0;
  )
  return 1;
}

int at_load_multi_(tensor *tensors, char **tensor_names, int ntensors, char *filename) {
  PROTECT(
    torch::NoGradGuard no_grad;
    torch::serialize::InputArchive archive;
    archive.load_from(std::string(filename));
    for (int i = 0; i < ntensors; ++i) {
      if (tensors[i]->device().type() == at::kCPU)
        archive.read(std::string(tensor_names[i]), *(tensors[i]));
      else {
        torch::Tensor tmp_tensor = torch::empty_like(*(tensors[i]), at::device(at::kCPU));
        archive.read(std::string(tensor_names[i]), tmp_tensor);
        tensors[i]->copy_(tmp_tensor);
      }
    }
    return 0;
  )
  return 1;
}

int at_load(tensor *out__, char *filename) {
  PROTECT(
    torch::Tensor tensor;
    torch::load(tensor, filename);
    out__[0] = new torch::Tensor(tensor);
    return 0;
  )
  return 1;
}

int at_get_num_interop_threads(int *out__) {
  PROTECT(
    out__[0] = at::get_num_interop_threads();
    return 0;
  )
  return 1;
}

int at_get_num_threads(int *out__) {
  PROTECT(
    out__[0] = at::get_num_threads();
    return 0;
  )
  return 1;
}

int at_set_num_interop_threads(int n_threads) {
  PROTECT(
    at::set_num_interop_threads(n_threads);
    return 0;
  )
  return 1;
}

int at_set_num_threads(int n_threads) {
  PROTECT(
    at::set_num_threads(n_threads);
    return 0;
  )
  return 1;
}

int at_set_qengine(int qengine_id) {
  PROTECT(
    at::QEngine qengine = at::QEngine::NoQEngine;
    switch (qengine_id) {
      case 0:
        break;
      case 1:
        qengine = at::QEngine::FBGEMM;
        break;
      case 2:
        qengine = at::QEngine::QNNPACK;
        break;
    }
    auto qengines = at::globalContext().supportedQEngines();
    if (std::find(qengines.begin(), qengines.end(), qengine) != qengines.end()) {
      at::globalContext().setQEngine(qengine);
    }
    else throw std::invalid_argument("unsupported qengine");
    return 0;
  )
  return 1;
}

int at_run_backward(tensor *tensors,
                     int ntensors,
                     tensor *inputs,
                     int ninputs,
                     tensor *outputs,
                     int keep_graph,
                     int create_graph) {
  PROTECT(
    vector<torch::autograd::Edge> roots;
    for (int i = 0; i < ntensors; ++i)
      roots.push_back(torch::autograd::impl::gradient_edge(*tensors[i]));

    vector<torch::autograd::Edge> inputs_;
    for (int i = 0; i < ninputs; ++i) {
      if (!inputs[i]->requires_grad())
        throw std::invalid_argument("one of the input tensor does not use set_requires_grad");
      inputs_.push_back(torch::autograd::impl::gradient_edge(*inputs[i]));
    }

    vector<torch::autograd::Variable> grads;
    for (int i = 0; i < ntensors; ++i)
      grads.push_back(torch::ones_like(*tensors[i]));

    auto vl = torch::autograd::Engine::get_default_engine().execute(roots, grads, keep_graph, create_graph, false, inputs_);
    for (int i = 0; i < ninputs; ++i) {
      outputs[i] = static_cast<tensor>(new torch::autograd::Variable(vl[i]));
    }
    return 0;
  )
  return 1;
}

int ato_adam(optimizer *out__, double learning_rate,
                   double beta1,
                   double beta2,
                   double weight_decay) {
  PROTECT(
    auto options =
      torch::optim::AdamOptions(learning_rate)
        .betas(std::tuple<double, double>(beta1, beta2))
        .weight_decay(weight_decay);
    out__[0] = new torch::optim::Adam(vector<torch::Tensor>(), options);
    return 0;
  )
  return 1;
}

int ato_adamw(optimizer *out__, double learning_rate,
                    double beta1,
                    double beta2,
                    double weight_decay) {
  PROTECT(
    auto options =
      torch::optim::AdamWOptions(learning_rate)
        .betas(std::tuple<double, double>(beta1, beta2))
        .weight_decay(weight_decay);
    out__[0] = new torch::optim::AdamW(vector<torch::Tensor>(), options);
    return 0;
  )
  return 1;
}

int ato_rms_prop(optimizer *out__, double learning_rate,
                       double alpha,
                       double eps,
                       double weight_decay,
                       double momentum,
                       int centered) {
  PROTECT(
    auto options =
      torch::optim::RMSpropOptions(learning_rate)
        .alpha(alpha)
        .eps(eps)
        .weight_decay(weight_decay)
        .momentum(momentum)
        .centered(centered != 0);
      out__[0] = new torch::optim::RMSprop(vector<torch::Tensor>(), options);
      return 0;
  )
  return 1;
}

int ato_sgd(optimizer *out__, double learning_rate,
                  double momentum,
                  double dampening,
                  double weight_decay,
                  int nesterov) {
  PROTECT(
    auto options =
      torch::optim::SGDOptions(learning_rate)
      .momentum(momentum)
      .dampening(dampening)
      .weight_decay(weight_decay)
      .nesterov(nesterov);
    out__[0] = new torch::optim::SGD(vector<torch::Tensor>(), options);
    return 0;
  )
  return 1;
}

int ato_add_parameters(optimizer t, tensor tensor, size_t group) {
  PROTECT(
    auto &groups = t->param_groups();
    while (groups.size() <= group) {
      groups.push_back(torch::optim::OptimizerParamGroup({}, t->defaults().clone()));
    }
    groups[group].params().push_back(*tensor);
    return 0;
  )
  return 1;
}

template <class T>
void set_lr(optimizer t, double learning_rate) {
  torch::optim::OptimizerOptions* d = &(t->defaults());
  if (auto p = dynamic_cast<T*>(d)) {
    p->lr(learning_rate);
    for (auto &param_group: t->param_groups()) {
      torch::optim::OptimizerOptions* d = &(param_group.options());
      if (auto p2 = dynamic_cast<T*>(d)) {
        p2->lr(learning_rate);
      }
      else throw std::invalid_argument("unexpected param group type");
    }
  }
}

int ato_set_learning_rate(optimizer t, double learning_rate) {
  PROTECT(
    set_lr<torch::optim::AdamOptions>(t, learning_rate);
    set_lr<torch::optim::AdamWOptions>(t, learning_rate);
    set_lr<torch::optim::RMSpropOptions>(t, learning_rate);
    set_lr<torch::optim::SGDOptions>(t, learning_rate);
    return 0;
  )
  return 1;
}

template <class T>
void set_lr_group(optimizer t, size_t group, double learning_rate) {
  auto &param_group = t->param_groups().at(group);
  torch::optim::OptimizerOptions* d = &(param_group.options());
  if (auto p = dynamic_cast<T*>(d)) {
    p->lr(learning_rate);
  }
}

int ato_set_learning_rate_group(optimizer t, size_t group, double learning_rate) {
  PROTECT(
    set_lr_group<torch::optim::AdamOptions>(t, group, learning_rate);
    set_lr_group<torch::optim::AdamWOptions>(t, group, learning_rate);
    set_lr_group<torch::optim::RMSpropOptions>(t, group, learning_rate);
    set_lr_group<torch::optim::SGDOptions>(t, group, learning_rate);
    return 0;
  )
  return 1;
}

int ato_set_momentum(optimizer t, double momentum) {
  PROTECT(
    torch::optim::OptimizerOptions* d = &(t->defaults());
    if (auto adam = dynamic_cast<torch::optim::AdamOptions*>(d)) {
      auto betas = adam->betas();
      adam->betas(std::tuple<double, double>(momentum, get<1>(betas)));
      for (auto &param_group: t->param_groups()) {
          torch::optim::OptimizerOptions* d = &(param_group.options());
          if (auto adam2 = dynamic_cast<torch::optim::AdamOptions*>(d)) {
              adam2->betas(std::tuple<double, double>(momentum, get<1>(betas)));
          }
          else throw std::invalid_argument("unexpected param group type");
      }
    }
    else if (auto adamw = dynamic_cast<torch::optim::AdamWOptions*>(d)) {
        auto betas = adamw->betas();
        adamw->betas(std::tuple<double, double>(momentum, get<1>(betas)));
        for (auto &param_group: t->param_groups()) {
            torch::optim::OptimizerOptions* d = &(param_group.options());
            if (auto adamw2 = dynamic_cast<torch::optim::AdamWOptions*>(d)) {
                adamw2->betas(std::tuple<double, double>(momentum, get<1>(betas)));
            }
            else throw std::invalid_argument("unexpected param group type");
        }
    }
    else if (auto rms = dynamic_cast<torch::optim::RMSpropOptions*>(d)) {
      rms->momentum(momentum);
      for (auto &param_group: t->param_groups()) {
          torch::optim::OptimizerOptions* d = &(param_group.options());
          if (auto rms2 = dynamic_cast<torch::optim::RMSpropOptions*>(d)) {
              rms2->momentum(momentum);
          }
          else throw std::invalid_argument("unexpected param group type");
      }
    }
    else if (auto sgd = dynamic_cast<torch::optim::SGDOptions*>(d)) {
      sgd->momentum(momentum);
      for (auto &param_group: t->param_groups()) {
          torch::optim::OptimizerOptions* d = &(param_group.options());
          if (auto sgd2 = dynamic_cast<torch::optim::SGDOptions*>(d)) {
              sgd2->momentum(momentum);
          }
          else throw std::invalid_argument("unexpected param group type");
      }
    }
    else
     throw std::invalid_argument("unexpected optimizer");
    return 0;
  )
  return 1;
}

int ato_set_momentum_group(optimizer t, size_t group, double momentum) {
  PROTECT(
    auto &param_group = t->param_groups().at(group);
    torch::optim::OptimizerOptions* d = &(param_group.options());

    if (auto adam = dynamic_cast<torch::optim::AdamOptions*>(d)) {
        auto betas = adam->betas();
        adam->betas(std::tuple<double, double>(momentum, get<1>(betas)));
    }
    else if (auto adamw = dynamic_cast<torch::optim::AdamWOptions*>(d)) {
        auto betas = adamw->betas();
        adamw->betas(std::tuple<double, double>(momentum, get<1>(betas)));
    }
    else if (auto rms = dynamic_cast<torch::optim::RMSpropOptions*>(d)) {
        rms->momentum(momentum);
    }
    if (auto sgd = dynamic_cast<torch::optim::SGDOptions*>(d)) {
        sgd->momentum(momentum);
    }
    else
        throw std::invalid_argument("unexpected optimizer");
    return 0;
  )
  return 1;
}

template <class T>
void set_weight_decay(optimizer t, double weight_decay) {
  torch::optim::OptimizerOptions* d = &(t->defaults());
  if (auto p = dynamic_cast<T*>(d)) {
    p->weight_decay(weight_decay);
    for (auto &param_group: t->param_groups()) {
      torch::optim::OptimizerOptions* d = &(param_group.options());
      if (auto p2 = dynamic_cast<T*>(d)) {
        p2->weight_decay(weight_decay);
      }
      else throw std::invalid_argument("unexpected param group type");
    }
  }
}

int ato_set_weight_decay(optimizer t, double weight_decay) {
  PROTECT(
    set_weight_decay<torch::optim::AdamOptions>(t, weight_decay);
    set_weight_decay<torch::optim::AdamWOptions>(t, weight_decay);
    set_weight_decay<torch::optim::RMSpropOptions>(t, weight_decay);
    set_weight_decay<torch::optim::SGDOptions>(t, weight_decay);
    return 0;
  )
  return 1;
}

template <class T>
void set_weight_decay_group(optimizer t, size_t group, double weight_decay) {
  auto &param_group = t->param_groups().at(group);
  torch::optim::OptimizerOptions* d = &(param_group.options());
  if (auto p = dynamic_cast<T*>(d)) {
    p->weight_decay(weight_decay);
  }
}

int ato_set_weight_decay_group(optimizer t, size_t group, double weight_decay) {
  PROTECT(
    set_weight_decay_group<torch::optim::AdamOptions>(t, group, weight_decay);
    set_weight_decay_group<torch::optim::AdamWOptions>(t, group, weight_decay);
    set_weight_decay_group<torch::optim::RMSpropOptions>(t, group, weight_decay);
    set_weight_decay_group<torch::optim::SGDOptions>(t, group, weight_decay);
    return 0;
  )
  return 1;
}

int ato_zero_grad(optimizer t) {
  PROTECT(
    t->zero_grad();
    return 0;
  )
  return 1;
}

int ato_step(optimizer t) {
  PROTECT(
    t->step();
    return 0;
  )
  return 1;
}

int ato_free(optimizer t) {
  PROTECT(
    delete(t);
    return 0;
  )
  return 1;
}

int ats_int(scalar *out__, int64_t v) {
  PROTECT(
    out__[0] = new torch::Scalar(v);
    return 0;
  )
  return 1;
}

int ats_float(scalar *out__, double v) {
  PROTECT(
    out__[0] = new torch::Scalar(v);
    return 0;
  )
  return 1;
}

int ats_to_int(int64_t *out__, scalar s) {
  PROTECT(
    out__[0] = s->toLong();
    return 0;
  )
  return 1;
}

int ats_to_float(double *out__, scalar s) {
  PROTECT(
    out__[0] = s->toDouble();
    return 0;
  )
  return 1;
}

int ats_to_string(char **out__, scalar s) {
  PROTECT(
    using namespace at;
    std::ostringstream oss;
    oss << (*s);
    out__[0] = strdup(oss.str().c_str());
    return 0;
  )
  return 1;
}

int ats_free(scalar s) {
  PROTECT(
    delete(s);
    return 0;
  )
  return 1;
}

int atc_cuda_device_count(int *out__) {
  PROTECT(
    out__[0] = torch::cuda::device_count();
    return 0;
  )
  return 1;
}

int atc_cuda_is_available(int *out__) {
  PROTECT(
    out__[0] = torch::cuda::is_available();
    return 0;
  )
  return 1;
}

int atc_cudnn_is_available(int *out__) {
  PROTECT(
    out__[0] = torch::cuda::cudnn_is_available();
    return 0;
  )
  return 1;
}

int atc_user_enabled_cudnn(int *out__) {
  PROTECT(
    out__[0] = at::globalContext().userEnabledCuDNN();
    return 0;
  )
  return 1;
}

int atc_set_user_enabled_cudnn(int b) {
  PROTECT(
    at::globalContext().setUserEnabledCuDNN(b);
    return 0;
  )
  return 1;
}

int atc_set_benchmark_cudnn(int b) {
  PROTECT(
    at::globalContext().setBenchmarkCuDNN(b);
    return 0;
  )
  return 1;
}

int atm_load(module *out__, char *filename) {
  PROTECT(
    out__[0] = new torch::jit::script::Module(torch::jit::load(filename));
    return 0;
  )
  return 1;
}

int atm_load_on_device(module *out__, char *filename, int device) {
  PROTECT(
    out__[0] = new torch::jit::script::Module(torch::jit::load(filename, device_of_int(device)));
    return 0;
  )
  return 1;
}

int atm_load_str(module *out__, char *data, size_t sz) {
  PROTECT(
    std::istringstream stream(std::string(data, sz));
    out__[0] = new torch::jit::script::Module(torch::jit::load(stream));
    return 0;
  )
  return 1;
}

int atm_load_str_on_device(module *out__, char *data, size_t sz, int device) {
  PROTECT(
    std::istringstream stream(std::string(data, sz));
    out__[0] = new torch::jit::script::Module(torch::jit::load(stream, device_of_int(device)));
    return 0;
  )
  return 1;
}

int atm_forward(tensor *out__, module m, tensor *tensors, int ntensors) {
  PROTECT(
    std::vector<torch::jit::IValue> inputs;
    for (int i = 0; i < ntensors; ++i)
      inputs.push_back(*(tensors[i]));
    torch::jit::IValue output = m->forward(std::move(inputs));
    if (!output.isTensor())
      throw std::invalid_argument("forward did not return a tensor");
    out__[0] = new torch::Tensor(output.toTensor());
    return 0;
  )
  return 1;
}

int atm_forward_(ivalue *out__, module m,
                    ivalue *ivalues,
                    int nivalues) {
  PROTECT(
    std::vector<torch::jit::IValue> inputs;
    for (int i = 0; i < nivalues; ++i)
      inputs.push_back(*(ivalues[i]));
    torch::jit::IValue output = m->forward(std::move(inputs));
    out__[0] = new torch::jit::IValue(output);
    return 0;
  )
  return 1;
}

int atm_method(tensor *out__, module m, char *method_name, tensor *tensors, int ntensors) {
  PROTECT(
    std::vector<torch::jit::IValue> inputs;
    for (int i = 0; i < ntensors; ++i)
      inputs.push_back(*(tensors[i]));
    torch::jit::IValue output = m->get_method(method_name)(std::move(inputs));
    if (!output.isTensor())
      throw std::invalid_argument("method did not return a tensor");
    out__[0] = new torch::Tensor(output.toTensor());
    return 0;
  )
  return 1;
}

int atm_method_(ivalue *out__, module m, char *method_name, ivalue *ivalues, int nivalues) {
  PROTECT(
    std::vector<torch::jit::IValue> inputs;
    for (int i = 0; i < nivalues; ++i)
      inputs.push_back(*(ivalues[i]));
    torch::jit::IValue output = m->get_method(method_name)(std::move(inputs));
    out__[0] = new torch::jit::IValue(output);
    return 0;
  )
  return 1;
}

int atm_eval(module m) {
  PROTECT(
    m->eval();
    return 0;
  )
  return 1;
}

int atm_train(module m) {
  PROTECT(
    m->train();
    return 0;
  )
  return 1;
}

int atm_free(module m) {
  PROTECT(
    delete(m);
    return 0;
  )
  return 1;
}

int atm_save(module m, char *filename) {
  PROTECT(
    m->save(filename);
    return 0;
  )
  return 1;
}

int atm_to(module m, int device, int dtype, bool non_blocking) {
  PROTECT(
    m->to(device_of_int(device), at::ScalarType(dtype), non_blocking);
    return 0;
  )
  return 1;
}

int atm_get_profiling_mode(int *out__) {
  PROTECT(
    return torch::jit::getProfilingMode();
  )
  return 0;
}

int atm_set_profiling_mode(int b) {
  PROTECT(
    torch::jit::getProfilingMode() = (bool)b;
    return 0;
  )
  return 1;
}

int atm_create_for_tracing(module *out__, 
    char *modl_name,
    tensor *inputs,
    int ninputs) {
  PROTECT(
    torch::jit::script::Module modl(modl_name);
    if (torch::jit::tracer::isTracing())
      throw std::invalid_argument("cannot nest tracing calls");
    auto state = std::make_shared<torch::jit::tracer::TracingState>();
    torch::jit::tracer::setTracingState(state);
    auto* _modl_value = state->graph->insertInput(0, "self")->setType(modl._ivalue()->type());
    for (int i = 0; i < ninputs; ++i) {
      auto value = state->graph->addInput();
      value->setType(torch::jit::TensorType::get());
      state->setValue(*inputs[i], value); 
    }
    out__[0] = new torch::jit::script::Module(modl);
    return 0;
  )
  torch::jit::tracer::abandon();
  return 1;
}

int atm_end_tracing(module m, char *fn_name, tensor *outputs, int noutputs) {
  PROTECT(
    auto state = torch::jit::tracer::getTracingState();
    if (state == nullptr)
      throw std::invalid_argument("not in tracing mode");
    for (int i = 0; i < noutputs; ++i) {
      state->graph->registerOutput(state->getOutput(*outputs[i], i));
    }
    torch::jit::FixupTraceScopeBlocks(state->graph, m);
    torch::jit::NormalizeOps(state->graph);
    torch::jit::tracer::setTracingState(nullptr);
    auto fn = m->_ivalue()->compilation_unit()->create_function(fn_name, state->graph);
    m->type()->addMethod(fn);
    return 0;
  )
  return 1;
}

int atm_named_parameters(module m, void *data, void (*f)(void *, char *, tensor)) {
  PROTECT(
    for (const auto &p : m->named_parameters()) {
      auto v = p.value;
      f(data, (char*)p.name.c_str(), new torch::Tensor(v));
    }
    return 0;
  )
  return 1;
}

int ati_tensor(ivalue *out__, tensor t) {
  PROTECT(
    out__[0] = new torch::jit::IValue(*t);
    return 0;
  )
  return 1;
}

int ati_int(ivalue *out__, int64_t i) {
  PROTECT(
    out__[0] = new torch::jit::IValue(i);
    return 0;
  )
  return 1;
}

int ati_double(ivalue *out__, double d) {
  PROTECT(
    out__[0] = new torch::jit::IValue(d);
    return 0;
  )
  return 1;
}

int ati_bool(ivalue *out__, int i) {
  PROTECT(
    out__[0] = new torch::jit::IValue((bool)i);
    return 0;
  )
  return 1;
}

int ati_string(ivalue *out__, char *s) {
  PROTECT(
    string str(s);
    out__[0] = new torch::jit::IValue(str);
    return 0;
  )
  return 1;
}

int ati_none(ivalue *out__) {
  PROTECT(
    out__[0] = new torch::jit::IValue();
    return 0;
  )
  return 1;
}

int ati_tuple(ivalue *out__, ivalue *is, int nvalues) {
  PROTECT(
    vector<torch::jit::IValue> vec;
    for (int i = 0; i < nvalues; ++i) vec.push_back(*(is[i]));
    out__[0] = new torch::jit::IValue(torch::ivalue::Tuple::create(vec));
    return 0;
  )
  return 1;
}

int ati_generic_list(ivalue *out__, ivalue *is, int nvalues) {
  PROTECT(
    c10::List<torch::jit::IValue> vec(c10::AnyType::get());
    for (int i = 0; i < nvalues; ++i) vec.push_back(*(is[i]));
    out__[0] = new torch::jit::IValue(c10::List<torch::jit::IValue>(vec));
    return 0;
  )
  return 1;
}

int ati_generic_dict(ivalue *out__, ivalue *is, int nvalues) {
  c10::Dict<torch::jit::IValue, torch::jit::IValue> dict(c10::AnyType::get(), c10::AnyType::get());
  PROTECT(
    for (int i = 0; i < nvalues; ++i) dict.insert(*(is[2*i]), *(is[2*i+1]));
    out__[0] = new torch::jit::IValue(dict);
    return 0;
  )
  return 1;
}

int ati_int_list(ivalue *out__, int64_t *is, int nvalues) {
  PROTECT(
    c10::List<int64_t> vec;
    for (int i = 0; i < nvalues; ++i) vec.push_back(is[i]);
    out__[0] = new torch::jit::IValue(vec);
    return 0;
  )
  return 1;
}

int ati_double_list(ivalue *out__, double *is, int nvalues) {
  PROTECT(
    c10::List<double> vec;
    for (int i = 0; i < nvalues; ++i) vec.push_back(is[i]);
    out__[0] = new torch::jit::IValue(vec);
    return 0;
  )
  return 1;
}

int ati_bool_list(ivalue *out__, char *is, int nvalues) {
  PROTECT(
    c10::List<bool> vec;
    for (int i = 0; i < nvalues; ++i) vec.push_back(is[i] != 0);
    out__[0] = new torch::jit::IValue(vec);
    return 0;
  )
  return 1;
}

int ati_string_list(ivalue *out__, char **is, int nvalues) {
  PROTECT(
    c10::List<string> vec;
    for (int i = 0; i < nvalues; ++i) vec.push_back(string(is[i]));
    out__[0] = new torch::jit::IValue(vec);
    return 0;
  )
  return 1;
}

int ati_tensor_list(ivalue *out__, tensor *is, int nvalues) {
  PROTECT(
    c10::List<at::Tensor> vec;
    for (int i = 0; i < nvalues; ++i) vec.push_back(*(is[i]));
    out__[0] = new torch::jit::IValue(vec);
    return 0;
  )
  return 1;
}

int ati_tag(int *out__, ivalue i) {
  PROTECT(
    if (i->isNone()) out__[0] = 0;
    else if (i->isTensor()) out__[0] = 1;
    else if (i->isDouble()) out__[0] = 2;
    else if (i->isInt()) out__[0] = 3;
    else if (i->isBool()) out__[0] = 4;
    else if (i->isTuple()) out__[0] = 5;
    else if (i->isIntList()) out__[0] = 6;
    else if (i->isDoubleList()) out__[0] = 7;
    else if (i->isBoolList()) out__[0] = 8;
    else if (i->isString()) out__[0] = 9;
    else if (i->isTensorList()) out__[0] = 10;
    else if (i->isList()) out__[0] = 12;
    else if (i->isGenericDict()) out__[0] = 13;
    else throw std::invalid_argument(("unsupported tag" + i->tagKind()).c_str());
    return 0;
  )
  return 1;
}

int ati_to_int(int64_t *out__, ivalue i) {
  PROTECT(
    out__[0] = i->toInt();
    return 0;
  )
  return 1;
}

int ati_to_double(double *out__, ivalue i) {
  PROTECT(
    out__[0] = i->toDouble();
    return 0;
  )
  return 1;
}

int ati_to_bool(int *out__, ivalue i) {
  PROTECT(
    out__[0] = i->toBool();
    return 0;
  )
  return 1;
}

int ati_to_string(char **out__, ivalue i) {
  PROTECT(
    auto str = i->toStringRef();
    out__[0] = strdup(str.c_str());
    return 0;
  )
  return 1;
}

int ati_to_tensor(tensor *out__, ivalue i) {
  PROTECT(
    out__[0] = new torch::Tensor(i->toTensor());
    return 0;
  )
  return 1;
}

int ati_length(int *out__, ivalue i) {
  PROTECT(
    if (i->isTuple()) out__[0] = i->toTuple()->elements().size();
    else if (i->isIntList()) out__[0] = i->toIntList().size();
    else if (i->isDoubleList()) out__[0] = i->toDoubleList().size();
    else if (i->isBoolList()) out__[0] = i->toBoolList().size();
    else if (i->isString()) out__[0] = i->toStringRef().size();
    else if (i->isTensorList()) out__[0] = i->toTensorList().size();
    else if (i->isList()) out__[0] = i->toList().size();
    else if (i->isGenericDict()) out__[0] = i->toGenericDict().size();
    else throw std::invalid_argument(("unsupported tag for length " + i->tagKind()).c_str());
    return 0;
  )
  return 1;
}

int ati_tuple_length(int *out__, ivalue i) {
  PROTECT(
    out__[0] = i->toTuple()->elements().size();
  )
  return -1;
}

int ati_to_tuple(ivalue i,
                  ivalue *outputs,
                  int noutputs) {
  PROTECT(
    auto vec = i->toTuple()->elements();
    if (vec.size() != noutputs) {
      throw std::invalid_argument("unexpected tuple size");
    }
    for (int i = 0; i < noutputs; ++i)
      outputs[i] = new torch::jit::IValue(vec[i]);
    return 0;
  )
  return 1;
}

int ati_to_generic_list(ivalue i,
                         ivalue *outputs,
                         int noutputs) {
  PROTECT(
    auto vec = i->toList();
    if (vec.size() != noutputs) {
      throw std::invalid_argument("unexpected list size");
    }
    for (int i = 0; i < noutputs; ++i)
      outputs[i] = new torch::jit::IValue(vec[i]);
    return 0;
  )
  return 1;
}

int ati_to_generic_dict(ivalue i,
                         ivalue *outputs,
                         int noutputs) {
  PROTECT(
    auto dict = i->toGenericDict();
    if (dict.size() != noutputs) {
      throw std::invalid_argument("unexpected dict size");
    }
    int k = 0;
    for (auto it = dict.begin(); it != dict.end(); ++it) {
      outputs[k++] = new torch::jit::IValue(it->key());
      outputs[k++] = new torch::jit::IValue(it->value());
    }
    return 0;
  )
  return 1;
}

int ati_to_int_list(ivalue i,
                  int64_t *outputs,
                  int noutputs) {
  PROTECT(
    auto vec = i->toIntList();
    if (vec.size() != noutputs) {
      throw std::invalid_argument("unexpected list<int> size");
    }
    for (int i = 0; i < noutputs; ++i)
      outputs[i] = vec[i];
    return 0;
  )
  return 1;
}

int ati_to_double_list(ivalue i,
                        double *outputs,
                        int noutputs) {
  PROTECT(
    auto vec = i->toDoubleList();
    if (vec.size() != noutputs) {
      throw std::invalid_argument("unexpected list<double> size");
    }
    for (int i = 0; i < noutputs; ++i)
      outputs[i] = vec[i];
    return 0;
  )
  return 1;
}

int ati_to_bool_list(ivalue i,
                      char *outputs,
                      int noutputs) {
  PROTECT(
    auto vec = i->toBoolList();
    if (vec.size() != noutputs) {
      throw std::invalid_argument("unexpected list<bool> size");
    }
    for (int i = 0; i < noutputs; ++i)
      outputs[i] = vec[i];
    return 0;
  )
  return 1;
}

int ati_to_tensor_list(ivalue i,
                        tensor *outputs,
                        int noutputs) {
  PROTECT(
    auto vec = i->toTensorList();
    if (vec.size() != noutputs) {
      throw std::invalid_argument("unexpected list<tensor> size");
    }
    for (int i = 0; i < noutputs; ++i)
      outputs[i] = new torch::Tensor(vec[i]);
    return 0;
  )
  return 1;
}


int ati_free(ivalue i) {
  PROTECT(
    delete(i);
    return 0;
  )
  return 1;
}

#include "torch_api_generated.cpp.h"
