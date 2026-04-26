// verigpu_backend.cpp — PyTorch custom backend for VeriGPU
// CP3: allocator, device guard, copy, empty, fill, zero
// CP4: add (Tensor, Scalar, in-place)
// CP5: sub, mul, div, neg, abs, relu, clamp + in-place variants
// CP6: sum, mean (full + dim), mm, addmm

#include <torch/extension.h>
#include <c10/core/impl/DeviceGuardImplInterface.h>
#include <cstring>
#include <cmath>
#include <algorithm>
#include <vector>
#include <numeric>

namespace {

// =====================================================================
// 1. ALLOCATOR
// =====================================================================

static void verigpu_delete(void* ptr) { free(ptr); }

struct VeriGPUAllocator final : public c10::Allocator {
    at::DataPtr allocate(size_t nbytes) override {
        void* data = nullptr;
        if (nbytes > 0) {
            data = malloc(nbytes);
            TORCH_CHECK(data, "VeriGPU: alloc failed for ", nbytes, " bytes");
            memset(data, 0, nbytes);
        }
        return {data, data, &verigpu_delete,
                at::Device(at::DeviceType::PrivateUse1, 0)};
    }
    at::DeleterFnPtr raw_deleter() const override { return &verigpu_delete; }
    void copy_data(void* dest, const void* src, std::size_t count)
        const override {
        if (dest != src && count > 0) memcpy(dest, src, count);
    }
};

static VeriGPUAllocator g_allocator;
static bool _reg = []() {
    c10::SetAllocator(c10::DeviceType::PrivateUse1, &g_allocator);
    return true;
}();

// =====================================================================
// 2. DEVICE GUARD
// =====================================================================

struct VeriGPUGuardImpl final : public c10::impl::DeviceGuardImplInterface {
    at::DeviceType type() const override { return at::DeviceType::PrivateUse1; }
    c10::Device exchangeDevice(c10::Device) const override {
        return c10::Device(at::DeviceType::PrivateUse1, 0); }
    c10::Device getDevice() const override {
        return c10::Device(at::DeviceType::PrivateUse1, 0); }
    void setDevice(c10::Device) const override {}
    void uncheckedSetDevice(c10::Device) const noexcept override {}
    c10::Stream getStream(c10::Device) const noexcept override {
        return c10::Stream(c10::Stream::DEFAULT,
                           c10::Device(at::DeviceType::PrivateUse1, 0)); }
    c10::Stream getDefaultStream(c10::Device) const override {
        return getStream(c10::Device(at::DeviceType::PrivateUse1, 0)); }
    c10::Stream exchangeStream(c10::Stream) const noexcept override {
        return c10::Stream(c10::Stream::DEFAULT,
                           c10::Device(at::DeviceType::PrivateUse1, 0)); }
    c10::DeviceIndex deviceCount() const noexcept override { return 1; }
    void record(void**, const c10::Stream&, const c10::DeviceIndex,
                const c10::EventFlag) const override {}
    void block(void*, const c10::Stream&) const override {}
    bool queryEvent(void*) const override { return true; }
    void destroyEvent(void*, const c10::DeviceIndex) const noexcept override {}
};

C10_REGISTER_GUARD_IMPL(PrivateUse1, VeriGPUGuardImpl);

// =====================================================================
// 3. HELPERS
// =====================================================================

static at::Tensor make_verigpu_tensor(
    at::IntArrayRef size, at::IntArrayRef stride, at::ScalarType dtype)
{
    int64_t nelements = 1;
    for (auto s : size) nelements *= s;
    size_t nbytes = static_cast<size_t>(nelements) * c10::elementSize(dtype);

    auto storage = c10::Storage(
        c10::Storage::use_byte_size_t(),
        static_cast<int64_t>(nbytes),
        g_allocator.allocate(nbytes), &g_allocator, true);

    auto tensor = at::detail::make_tensor<c10::TensorImpl>(
        std::move(storage),
        c10::DispatchKeySet(c10::DispatchKey::PrivateUse1),
        at::scalarTypeToTypeMeta(dtype));

    tensor.unsafeGetTensorImpl()->set_sizes_and_strides(size, stride);
    return tensor;
}

// Create contiguous tensor with given shape and dtype
static at::Tensor make_verigpu_contiguous(
    at::IntArrayRef size, at::ScalarType dtype)
{
    std::vector<int64_t> strides(size.size());
    if (!size.empty()) {
        strides.back() = 1;
        for (int64_t i = (int64_t)size.size() - 2; i >= 0; i--)
            strides[i] = strides[i + 1] * size[i + 1];
    }
    return make_verigpu_tensor(size, strides, dtype);
}

static at::Tensor make_output_like(const at::Tensor& ref) {
    return make_verigpu_contiguous(ref.sizes(), ref.scalar_type());
}

struct BinaryArgs { at::Tensor a, b; bool b_is_scalar; };
static BinaryArgs prepare_binary(const at::Tensor& self, const at::Tensor& other) {
    auto a = self.contiguous();
    auto b = other.contiguous();
    if (b.dim() == 0 && b.scalar_type() != a.scalar_type()) b = b.to(a.scalar_type());
    if (a.dim() == 0 && a.scalar_type() != b.scalar_type()) a = a.to(b.scalar_type());

    bool bs = (b.dim() == 0);

    if (!bs && a.sizes() != b.sizes()) {
        // Broadcasting via CPU (expand is not registered on our backend).
        // Since our "GPU memory" is host memory, the CPU roundtrip is free.
        auto a_cpu = a.cpu();
        auto b_cpu = b.cpu();
        auto target_dev = at::Device(at::DeviceType::PrivateUse1, 0);
        try {
            b_cpu = b_cpu.expand(a_cpu.sizes()).contiguous();
            b = b_cpu.to(target_dev);
        } catch (...) {
            try {
                a_cpu = a_cpu.expand(b_cpu.sizes()).contiguous();
                a = a_cpu.to(target_dev);
            } catch (...) {
                TORCH_CHECK(false,
                    "VeriGPU: shape mismatch (", a.sizes(), " vs ", b.sizes(), ")");
            }
        }
    }
    TORCH_CHECK(a.scalar_type() == b.scalar_type(), "VeriGPU: dtype mismatch");
    return {a, b, bs};
}

template <typename F>
static at::Tensor binary_op(const at::Tensor& self, const at::Tensor& other, F op) {
    auto [a, b, b_scalar] = prepare_binary(self, other);
    auto output = make_output_like(a);
    auto n = a.numel();
    auto dtype = a.scalar_type();
    #define VERIGPU_BINARY_LOOP(T) { \
        const T* pa = a.data_ptr<T>(); const T* pb = b.data_ptr<T>(); T* po = output.data_ptr<T>(); \
        T bv = b_scalar ? pb[0] : T(0); \
        for (int64_t i = 0; i < n; i++) po[i] = op(pa[i], b_scalar ? bv : pb[i]); }
    if      (dtype == at::ScalarType::Float)  VERIGPU_BINARY_LOOP(float)
    else if (dtype == at::ScalarType::Double) VERIGPU_BINARY_LOOP(double)
    else if (dtype == at::ScalarType::Int)    VERIGPU_BINARY_LOOP(int32_t)
    else if (dtype == at::ScalarType::Long)   VERIGPU_BINARY_LOOP(int64_t)
    else TORCH_CHECK(false, "VeriGPU: unsupported dtype ", dtype);
    #undef VERIGPU_BINARY_LOOP
    return output;
}

template <typename F>
static at::Tensor& binary_op_inplace(at::Tensor& self, const at::Tensor& other, F op) {
    auto b = other.contiguous();
    if (b.dim() == 0 && b.scalar_type() != self.scalar_type()) b = b.to(self.scalar_type());
    bool b_scalar = (b.dim() == 0);
    auto n = self.numel();
    auto dtype = self.scalar_type();
    #define VERIGPU_INPLACE_LOOP(T) { \
        T* pa = self.data_ptr<T>(); const T* pb = b.data_ptr<T>(); \
        T bv = b_scalar ? pb[0] : T(0); \
        for (int64_t i = 0; i < n; i++) pa[i] = op(pa[i], b_scalar ? bv : pb[i]); }
    if      (dtype == at::ScalarType::Float)  VERIGPU_INPLACE_LOOP(float)
    else if (dtype == at::ScalarType::Double) VERIGPU_INPLACE_LOOP(double)
    else if (dtype == at::ScalarType::Int)    VERIGPU_INPLACE_LOOP(int32_t)
    else if (dtype == at::ScalarType::Long)   VERIGPU_INPLACE_LOOP(int64_t)
    else TORCH_CHECK(false, "VeriGPU: unsupported dtype ", dtype);
    #undef VERIGPU_INPLACE_LOOP
    return self;
}

template <typename F>
static at::Tensor unary_op(const at::Tensor& self, F op) {
    auto a = self.contiguous();
    auto output = make_output_like(a);
    auto n = a.numel();
    auto dtype = a.scalar_type();
    #define VERIGPU_UNARY_LOOP(T) { \
        const T* pa = a.data_ptr<T>(); T* po = output.data_ptr<T>(); \
        for (int64_t i = 0; i < n; i++) po[i] = op(pa[i]); }
    if      (dtype == at::ScalarType::Float)  VERIGPU_UNARY_LOOP(float)
    else if (dtype == at::ScalarType::Double) VERIGPU_UNARY_LOOP(double)
    else if (dtype == at::ScalarType::Int)    VERIGPU_UNARY_LOOP(int32_t)
    else if (dtype == at::ScalarType::Long)   VERIGPU_UNARY_LOOP(int64_t)
    else TORCH_CHECK(false, "VeriGPU: unsupported dtype ", dtype);
    #undef VERIGPU_UNARY_LOOP
    return output;
}

// =====================================================================
// 4. INFRASTRUCTURE OPS
// =====================================================================

at::Tensor verigpu_empty(
    at::IntArrayRef size, std::optional<at::ScalarType> dtype_opt,
    std::optional<at::Layout>, std::optional<at::Device>,
    std::optional<bool>, std::optional<at::MemoryFormat> fmt_opt) {
    auto dtype = dtype_opt.value_or(at::ScalarType::Float);
    auto t = make_verigpu_contiguous(size, dtype);
    if (fmt_opt.has_value())
        t.unsafeGetTensorImpl()->empty_tensor_restride(*fmt_opt);
    return t;
}

at::Tensor verigpu_empty_strided(
    at::IntArrayRef size, at::IntArrayRef stride,
    std::optional<at::ScalarType> dtype_opt, std::optional<at::Layout>,
    std::optional<at::Device>, std::optional<bool>) {
    return make_verigpu_tensor(size, stride,
        dtype_opt.value_or(at::ScalarType::Float));
}

at::Tensor& verigpu_copy_(at::Tensor& self, const at::Tensor& src, bool) {
    auto nbytes = self.nbytes();
    if (nbytes > 0 && self.data_ptr() != src.data_ptr()) {
        auto sc = src.contiguous();
        TORCH_CHECK(nbytes == (size_t)sc.nbytes(), "VeriGPU copy_: size mismatch");
        memcpy(self.data_ptr(), sc.data_ptr(), nbytes);
    }
    return self;
}
at::Tensor verigpu_copy_from(const at::Tensor& self, const at::Tensor& dst, bool) {
    if (self.nbytes() > 0) { auto sc = self.contiguous(); memcpy(dst.data_ptr(), sc.data_ptr(), sc.nbytes()); }
    return dst;
}
at::Tensor verigpu_copy_from_and_resize(const at::Tensor& self, const at::Tensor& dst) {
    dst.resize_as_(self);
    if (self.nbytes() > 0) { auto sc = self.contiguous(); memcpy(dst.data_ptr(), sc.data_ptr(), sc.nbytes()); }
    return dst;
}

at::Tensor& verigpu_fill_scalar(at::Tensor& self, const at::Scalar& value) {
    auto n = self.numel(); void* ptr = self.data_ptr(); auto dtype = self.scalar_type();
    if      (dtype == at::ScalarType::Float)  { float v=value.toFloat();   for(int64_t i=0;i<n;i++) static_cast<float*>(ptr)[i]=v; }
    else if (dtype == at::ScalarType::Double) { double v=value.toDouble(); for(int64_t i=0;i<n;i++) static_cast<double*>(ptr)[i]=v; }
    else if (dtype == at::ScalarType::Int)    { int32_t v=value.toInt();   for(int64_t i=0;i<n;i++) static_cast<int32_t*>(ptr)[i]=v; }
    else if (dtype == at::ScalarType::Long)   { int64_t v=value.toLong();  for(int64_t i=0;i<n;i++) static_cast<int64_t*>(ptr)[i]=v; }
    else if (dtype == at::ScalarType::Bool)   { bool v=value.toBool();     for(int64_t i=0;i<n;i++) static_cast<bool*>(ptr)[i]=v; }
    else { auto cpu=self.to(at::kCPU); cpu.fill_(value); memcpy(ptr,cpu.data_ptr(),self.nbytes()); }
    return self;
}

at::Tensor& verigpu_zero_(at::Tensor& self) {
    if (self.nbytes() > 0) memset(self.data_ptr(), 0, self.nbytes());
    return self;
}

// =====================================================================
// 5. ADD
// =====================================================================

at::Tensor verigpu_add_tensor(const at::Tensor& self, const at::Tensor& other, const at::Scalar& alpha) {
    auto [a, b, b_scalar] = prepare_binary(self, other);
    auto output = make_output_like(a); auto n = a.numel(); auto dtype = a.scalar_type();
    #define VERIGPU_ADD_LOOP(T) { const T* pa=a.data_ptr<T>(); const T* pb=b.data_ptr<T>(); T* po=output.data_ptr<T>(); \
        T av=alpha.to<T>(); T bv=b_scalar?pb[0]:T(0); for(int64_t i=0;i<n;i++) po[i]=pa[i]+av*(b_scalar?bv:pb[i]); }
    if      (dtype == at::ScalarType::Float)  VERIGPU_ADD_LOOP(float)
    else if (dtype == at::ScalarType::Double) VERIGPU_ADD_LOOP(double)
    else if (dtype == at::ScalarType::Int)    VERIGPU_ADD_LOOP(int32_t)
    else if (dtype == at::ScalarType::Long)   VERIGPU_ADD_LOOP(int64_t)
    else TORCH_CHECK(false, "VeriGPU add: unsupported dtype");
    #undef VERIGPU_ADD_LOOP
    return output;
}
at::Tensor& verigpu_add_tensor_(at::Tensor& self, const at::Tensor& other, const at::Scalar& alpha) {
    auto b = other.contiguous();
    if (b.dim()==0 && b.scalar_type()!=self.scalar_type()) b=b.to(self.scalar_type());
    bool bs=(b.dim()==0); auto n=self.numel(); auto dtype=self.scalar_type();
    #define VERIGPU_ADDI_LOOP(T) { T* pa=self.data_ptr<T>(); const T* pb=b.data_ptr<T>(); \
        T av=alpha.to<T>(); T bv=bs?pb[0]:T(0); for(int64_t i=0;i<n;i++) pa[i]+=av*(bs?bv:pb[i]); }
    if      (dtype == at::ScalarType::Float)  VERIGPU_ADDI_LOOP(float)
    else if (dtype == at::ScalarType::Double) VERIGPU_ADDI_LOOP(double)
    else if (dtype == at::ScalarType::Int)    VERIGPU_ADDI_LOOP(int32_t)
    else if (dtype == at::ScalarType::Long)   VERIGPU_ADDI_LOOP(int64_t)
    else TORCH_CHECK(false, "VeriGPU add_: unsupported dtype");
    #undef VERIGPU_ADDI_LOOP
    return self;
}
at::Tensor verigpu_add_scalar(const at::Tensor& self, const at::Scalar& other, const at::Scalar& alpha) {
    return unary_op(self, [v=alpha.toDouble()*other.toDouble()](auto x){ return decltype(x)(x+v); });
}

// =====================================================================
// 6. SUB
// =====================================================================

at::Tensor verigpu_sub_tensor(const at::Tensor& self, const at::Tensor& other, const at::Scalar& alpha) {
    auto [a, b, b_scalar] = prepare_binary(self, other);
    auto output = make_output_like(a); auto n = a.numel(); auto dtype = a.scalar_type();
    #define VERIGPU_SUB_LOOP(T) { const T* pa=a.data_ptr<T>(); const T* pb=b.data_ptr<T>(); T* po=output.data_ptr<T>(); \
        T av=alpha.to<T>(); T bv=b_scalar?pb[0]:T(0); for(int64_t i=0;i<n;i++) po[i]=pa[i]-av*(b_scalar?bv:pb[i]); }
    if      (dtype == at::ScalarType::Float)  VERIGPU_SUB_LOOP(float)
    else if (dtype == at::ScalarType::Double) VERIGPU_SUB_LOOP(double)
    else if (dtype == at::ScalarType::Int)    VERIGPU_SUB_LOOP(int32_t)
    else if (dtype == at::ScalarType::Long)   VERIGPU_SUB_LOOP(int64_t)
    else TORCH_CHECK(false, "VeriGPU sub: unsupported dtype");
    #undef VERIGPU_SUB_LOOP
    return output;
}
at::Tensor& verigpu_sub_tensor_(at::Tensor& self, const at::Tensor& other, const at::Scalar& alpha) {
    auto b = other.contiguous();
    if (b.dim()==0 && b.scalar_type()!=self.scalar_type()) b=b.to(self.scalar_type());
    bool bs=(b.dim()==0); auto n=self.numel(); auto dtype=self.scalar_type();
    #define VERIGPU_SUBI_LOOP(T) { T* pa=self.data_ptr<T>(); const T* pb=b.data_ptr<T>(); \
        T av=alpha.to<T>(); T bv=bs?pb[0]:T(0); for(int64_t i=0;i<n;i++) pa[i]-=av*(bs?bv:pb[i]); }
    if      (dtype == at::ScalarType::Float)  VERIGPU_SUBI_LOOP(float)
    else if (dtype == at::ScalarType::Double) VERIGPU_SUBI_LOOP(double)
    else if (dtype == at::ScalarType::Int)    VERIGPU_SUBI_LOOP(int32_t)
    else if (dtype == at::ScalarType::Long)   VERIGPU_SUBI_LOOP(int64_t)
    else TORCH_CHECK(false, "VeriGPU sub_: unsupported dtype");
    #undef VERIGPU_SUBI_LOOP
    return self;
}
at::Tensor verigpu_sub_scalar(const at::Tensor& self, const at::Scalar& other, const at::Scalar& alpha) {
    return unary_op(self, [v=alpha.toDouble()*other.toDouble()](auto x){ return decltype(x)(x-v); });
}

// =====================================================================
// 7. MUL 
// =====================================================================

at::Tensor verigpu_mul_tensor(const at::Tensor& self, const at::Tensor& other) {
    return binary_op(self, other, [](auto a, auto b){ return a*b; }); }
at::Tensor& verigpu_mul_tensor_(at::Tensor& self, const at::Tensor& other) {
    return binary_op_inplace(self, other, [](auto a, auto b){ return a*b; }); }
at::Tensor verigpu_mul_scalar(const at::Tensor& self, const at::Scalar& other) {
    return unary_op(self, [v=other.toDouble()](auto x){ return decltype(x)(x*v); }); }

// =====================================================================
// 8. DIV
// =====================================================================

at::Tensor verigpu_div_tensor(const at::Tensor& self, const at::Tensor& other) {
    return binary_op(self, other, [](auto a, auto b){ return a/b; }); }
at::Tensor& verigpu_div_tensor_(at::Tensor& self, const at::Tensor& other) {
    return binary_op_inplace(self, other, [](auto a, auto b){ return a/b; }); }
at::Tensor verigpu_div_scalar(const at::Tensor& self, const at::Scalar& other) {
    return unary_op(self, [v=other.toDouble()](auto x){ return decltype(x)(x/v); }); }

// =====================================================================
// 9. UNARY OPS
// =====================================================================

at::Tensor verigpu_neg(const at::Tensor& self) {
    return unary_op(self, [](auto x){ return -x; }); }
at::Tensor verigpu_abs(const at::Tensor& self) {
    return unary_op(self, [](auto x){ return x<0?-x:x; }); }
at::Tensor verigpu_relu(const at::Tensor& self) {
    return unary_op(self, [](auto x){ return x>0?x:decltype(x)(0); }); }
at::Tensor& verigpu_relu_(at::Tensor& self) {
    auto n=self.numel(); auto dtype=self.scalar_type();
    if (dtype==at::ScalarType::Float) { float* p=self.data_ptr<float>(); for(int64_t i=0;i<n;i++) if(p[i]<0)p[i]=0; }
    else if (dtype==at::ScalarType::Double) { double* p=self.data_ptr<double>(); for(int64_t i=0;i<n;i++) if(p[i]<0)p[i]=0; }
    return self;
}

at::Tensor verigpu_clamp(const at::Tensor& self,
    const std::optional<at::Scalar>& min_val, const std::optional<at::Scalar>& max_val) {
    auto a = self.contiguous(); auto output = make_output_like(a);
    auto n = a.numel(); auto dtype = a.scalar_type();
    if (dtype == at::ScalarType::Float) {
        const float* pa=a.data_ptr<float>(); float* po=output.data_ptr<float>();
        float lo = min_val.has_value() ? min_val->toFloat() : -INFINITY;
        float hi = max_val.has_value() ? max_val->toFloat() : INFINITY;
        for (int64_t i=0;i<n;i++) po[i]=std::max(lo, std::min(hi, pa[i]));
    } else if (dtype == at::ScalarType::Double) {
        const double* pa=a.data_ptr<double>(); double* po=output.data_ptr<double>();
        double lo = min_val.has_value() ? min_val->toDouble() : -INFINITY;
        double hi = max_val.has_value() ? max_val->toDouble() : INFINITY;
        for (int64_t i=0;i<n;i++) po[i]=std::max(lo, std::min(hi, pa[i]));
    } else TORCH_CHECK(false, "VeriGPU clamp: unsupported dtype");
    return output;
}

// =====================================================================
// 10. SUM — full reduction and reduction along dimensions
// =====================================================================
//
// sum reduces a tensor to a scalar (full) or reduces specific dimensions.
//
// Full sum: tensor([1,2,3,4]).sum() → tensor(10)
//   Simply iterate all elements and accumulate.
//
// Dim sum: tensor([[1,2],[3,4]]).sum(dim=0) → tensor([4,6])
//   For a [R,C] tensor with dim=0: output[j] = Σ_i input[i,j]
//   For a [R,C] tensor with dim=1: output[i] = Σ_j input[i,j]
//
// The general N-dimensional case uses stride arithmetic:
//   The reduced dimension is "collapsed" — we iterate over all
//   positions in the output, and for each, sum across the reduced dim.

// Full reduction: sum all elements to a scalar
at::Tensor verigpu_sum(const at::Tensor& self, std::optional<at::ScalarType> dtype_opt) {
    auto a = self.contiguous();
    auto out_dtype = dtype_opt.value_or(a.scalar_type());
    auto output = make_verigpu_contiguous({}, out_dtype);  // 0-dim tensor
    auto n = a.numel();

    if (a.scalar_type() == at::ScalarType::Float) {
        const float* pa = a.data_ptr<float>();
        double acc = 0;  // accumulate in double for precision
        for (int64_t i = 0; i < n; i++) acc += pa[i];
        if (out_dtype == at::ScalarType::Float)
            *output.data_ptr<float>() = static_cast<float>(acc);
        else
            *output.data_ptr<double>() = acc;
    } else if (a.scalar_type() == at::ScalarType::Double) {
        const double* pa = a.data_ptr<double>();
        double acc = 0;
        for (int64_t i = 0; i < n; i++) acc += pa[i];
        *output.data_ptr<double>() = acc;
    } else if (a.scalar_type() == at::ScalarType::Int) {
        const int32_t* pa = a.data_ptr<int32_t>();
        int64_t acc = 0;
        for (int64_t i = 0; i < n; i++) acc += pa[i];
        if (out_dtype == at::ScalarType::Long)
            *output.data_ptr<int64_t>() = acc;
        else
            *output.data_ptr<int32_t>() = static_cast<int32_t>(acc);
    } else if (a.scalar_type() == at::ScalarType::Long) {
        const int64_t* pa = a.data_ptr<int64_t>();
        int64_t acc = 0;
        for (int64_t i = 0; i < n; i++) acc += pa[i];
        *output.data_ptr<int64_t>() = acc;
    } else {
        TORCH_CHECK(false, "VeriGPU sum: unsupported dtype");
    }
    return output;
}

// Reduction along specific dimensions
at::Tensor verigpu_sum_dim(const at::Tensor& self,
    at::OptionalIntArrayRef dim_opt, bool keepdim,
    std::optional<at::ScalarType> dtype_opt)
{
    auto a = self.contiguous();
    auto out_dtype = dtype_opt.value_or(a.scalar_type());

    // If no dims specified, reduce all
    if (!dim_opt.has_value() || dim_opt->empty()) {
        auto result = verigpu_sum(self, dtype_opt);
        if (keepdim) {
            std::vector<int64_t> shape(a.dim(), 1);
            result = result.reshape(shape);
        }
        return result;
    }

    auto dims = dim_opt->vec();
    int64_t ndim = a.dim();

    // Normalize negative dims
    for (auto& d : dims) {
        if (d < 0) d += ndim;
        TORCH_CHECK(d >= 0 && d < ndim, "VeriGPU sum: dim out of range");
    }
    std::sort(dims.begin(), dims.end());

    // Compute output shape
    std::vector<int64_t> out_shape;
    for (int64_t i = 0; i < ndim; i++) {
        bool is_reduced = std::find(dims.begin(), dims.end(), i) != dims.end();
        if (is_reduced) {
            if (keepdim) out_shape.push_back(1);
        } else {
            out_shape.push_back(a.size(i));
        }
    }

    auto output = make_verigpu_contiguous(out_shape, out_dtype);

    // For simplicity and correctness: move to CPU, reduce, move back.
    // This handles arbitrary dim combinations correctly.
    // Performance is not a concern (host memory anyway).
    auto cpu_result = a.cpu().sum(dim_opt, keepdim, dtype_opt);
    TORCH_CHECK((int64_t)cpu_result.nbytes() == (int64_t)output.nbytes(),
        "VeriGPU sum_dim: size mismatch after reduction");
    memcpy(output.data_ptr(), cpu_result.data_ptr(), cpu_result.nbytes());

    return output;
}

// =====================================================================
// 11. MEAN — full and along dimensions
// =====================================================================

at::Tensor verigpu_mean(const at::Tensor& self, std::optional<at::ScalarType> dtype_opt) {
    auto s = verigpu_sum(self, dtype_opt);
    auto n = self.numel();
    if (s.scalar_type() == at::ScalarType::Float)
        *s.data_ptr<float>() /= n;
    else if (s.scalar_type() == at::ScalarType::Double)
        *s.data_ptr<double>() /= n;
    else
        TORCH_CHECK(false, "VeriGPU mean: only float/double supported");
    return s;
}

at::Tensor verigpu_mean_dim(const at::Tensor& self,
    at::OptionalIntArrayRef dim_opt, bool keepdim,
    std::optional<at::ScalarType> dtype_opt)
{
    // Use CPU for dim reduction (same rationale as sum_dim)
    auto a = self.contiguous();
    auto out_dtype = dtype_opt.value_or(a.scalar_type());

    auto cpu_result = a.cpu().mean(dim_opt, keepdim, dtype_opt);

    std::vector<int64_t> out_shape(cpu_result.sizes().begin(), cpu_result.sizes().end());
    auto output = make_verigpu_contiguous(out_shape, cpu_result.scalar_type());
    memcpy(output.data_ptr(), cpu_result.data_ptr(), cpu_result.nbytes());
    return output;
}

// =====================================================================
// 12. MM — Matrix multiplication
// =====================================================================
//
// mm(A, B): A is [M, K], B is [K, N] → result is [M, N]
// C[i][j] = Σ_k A[i][k] * B[k][j]
//
// This is the naive O(M*N*K) algorithm. For the small matrices we
// deal with in simulation, this is perfectly adequate.

at::Tensor verigpu_mm(const at::Tensor& self, const at::Tensor& mat2) {
    auto a = self.contiguous();
    auto b = mat2.contiguous();

    TORCH_CHECK(a.dim() == 2, "VeriGPU mm: self must be 2D, got ", a.dim());
    TORCH_CHECK(b.dim() == 2, "VeriGPU mm: mat2 must be 2D, got ", b.dim());
    TORCH_CHECK(a.size(1) == b.size(0),
        "VeriGPU mm: inner dimensions mismatch (", a.size(1), " vs ", b.size(0), ")");
    TORCH_CHECK(a.scalar_type() == b.scalar_type(), "VeriGPU mm: dtype mismatch");

    int64_t M = a.size(0), K = a.size(1), N = b.size(1);
    auto output = make_verigpu_contiguous({M, N}, a.scalar_type());

    auto dtype = a.scalar_type();
    if (dtype == at::ScalarType::Float) {
        const float* pa = a.data_ptr<float>();
        const float* pb = b.data_ptr<float>();
        float* po = output.data_ptr<float>();
        for (int64_t i = 0; i < M; i++) {
            for (int64_t j = 0; j < N; j++) {
                double acc = 0;  // accumulate in double
                for (int64_t k = 0; k < K; k++)
                    acc += (double)pa[i * K + k] * (double)pb[k * N + j];
                po[i * N + j] = static_cast<float>(acc);
            }
        }
    } else if (dtype == at::ScalarType::Double) {
        const double* pa = a.data_ptr<double>();
        const double* pb = b.data_ptr<double>();
        double* po = output.data_ptr<double>();
        for (int64_t i = 0; i < M; i++) {
            for (int64_t j = 0; j < N; j++) {
                double acc = 0;
                for (int64_t k = 0; k < K; k++)
                    acc += pa[i * K + k] * pb[k * N + j];
                po[i * N + j] = acc;
            }
        }
    } else {
        TORCH_CHECK(false, "VeriGPU mm: only float/double supported");
    }

    return output;
}

// =====================================================================
// 13. ADDMM — bias + matmul: out = beta*self + alpha*(mat1 @ mat2)
// =====================================================================
// Used heavily by nn.Linear: output = input @ weight.T + bias
// addmm(bias, input, weight, beta=1, alpha=1)

at::Tensor verigpu_addmm(const at::Tensor& self, const at::Tensor& mat1,
    const at::Tensor& mat2, const at::Scalar& beta, const at::Scalar& alpha)
{
    auto a = mat1.contiguous();
    auto b = mat2.contiguous();
    auto bias = self.contiguous();

    TORCH_CHECK(a.dim() == 2 && b.dim() == 2, "VeriGPU addmm: inputs must be 2D");
    TORCH_CHECK(a.size(1) == b.size(0), "VeriGPU addmm: inner dim mismatch");

    int64_t M = a.size(0), K = a.size(1), N = b.size(1);
    auto dtype = a.scalar_type();
    auto output = make_verigpu_contiguous({M, N}, dtype);

    if (dtype == at::ScalarType::Float) {
        const float* pa = a.data_ptr<float>();
        const float* pb = b.data_ptr<float>();
        const float* pc = bias.data_ptr<float>();
        float* po = output.data_ptr<float>();
        float alpha_v = alpha.toFloat();
        float beta_v = beta.toFloat();

        // bias can be 1D [N] (broadcast over rows) or 2D [M, N]
        bool bias_1d = (bias.dim() == 1);

        for (int64_t i = 0; i < M; i++) {
            for (int64_t j = 0; j < N; j++) {
                double acc = 0;
                for (int64_t k = 0; k < K; k++)
                    acc += (double)pa[i * K + k] * (double)pb[k * N + j];
                float bias_val = bias_1d ? pc[j] : pc[i * N + j];
                po[i * N + j] = beta_v * bias_val + alpha_v * static_cast<float>(acc);
            }
        }
    } else if (dtype == at::ScalarType::Double) {
        const double* pa = a.data_ptr<double>();
        const double* pb = b.data_ptr<double>();
        const double* pc = bias.data_ptr<double>();
        double* po = output.data_ptr<double>();
        double alpha_v = alpha.toDouble();
        double beta_v = beta.toDouble();
        bool bias_1d = (bias.dim() == 1);

        for (int64_t i = 0; i < M; i++) {
            for (int64_t j = 0; j < N; j++) {
                double acc = 0;
                for (int64_t k = 0; k < K; k++)
                    acc += pa[i * K + k] * pb[k * N + j];
                double bias_val = bias_1d ? pc[j] : pc[i * N + j];
                po[i * N + j] = beta_v * bias_val + alpha_v * acc;
            }
        }
    } else {
        TORCH_CHECK(false, "VeriGPU addmm: only float/double supported");
    }

    return output;
}

// =====================================================================
// 14. REGISTER ALL OPERATIONS
// =====================================================================

TORCH_LIBRARY_IMPL(aten, PrivateUse1, m) {
    // CP-3: infrastructure
    m.impl("empty.memory_format",    &verigpu_empty);
    m.impl("empty_strided",          &verigpu_empty_strided);
    m.impl("copy_",                  &verigpu_copy_);
    m.impl("_copy_from",             &verigpu_copy_from);
    m.impl("_copy_from_and_resize",  &verigpu_copy_from_and_resize);
    m.impl("fill_.Scalar",           &verigpu_fill_scalar);
    m.impl("zero_",                  &verigpu_zero_);

    // CP-4: add
    m.impl("add.Tensor",             &verigpu_add_tensor);
    m.impl("add_.Tensor",            &verigpu_add_tensor_);
    m.impl("add.Scalar",             &verigpu_add_scalar);

    // CP-5: sub, mul, div, unary
    m.impl("sub.Tensor",             &verigpu_sub_tensor);
    m.impl("sub_.Tensor",            &verigpu_sub_tensor_);
    m.impl("sub.Scalar",             &verigpu_sub_scalar);
    m.impl("mul.Tensor",             &verigpu_mul_tensor);
    m.impl("mul_.Tensor",            &verigpu_mul_tensor_);
    m.impl("mul.Scalar",             &verigpu_mul_scalar);
    m.impl("div.Tensor",             &verigpu_div_tensor);
    m.impl("div_.Tensor",            &verigpu_div_tensor_);
    m.impl("div.Scalar",             &verigpu_div_scalar);
    m.impl("neg",                    &verigpu_neg);
    m.impl("abs",                    &verigpu_abs);
    m.impl("relu",                   &verigpu_relu);
    m.impl("relu_",                  &verigpu_relu_);
    m.impl("clamp",                  &verigpu_clamp);

    // CP-6: reduction + matmul
    m.impl("sum",                    &verigpu_sum);
    m.impl("sum.dim_IntList",        &verigpu_sum_dim);
    m.impl("mean",                   &verigpu_mean);
    m.impl("mean.dim",               &verigpu_mean_dim);
    m.impl("mm",                     &verigpu_mm);
    m.impl("addmm",                  &verigpu_addmm);
}

} // anonymous namespace

// =====================================================================
// PYTHON BINDINGS
// =====================================================================

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.doc() = "VeriGPU backend for PyTorch";
    m.def("is_available", []() -> bool { return true; });
    m.def("device_count", []() -> int { return 1; });
    m.def("current_device", []() -> int { return 0; });
}
