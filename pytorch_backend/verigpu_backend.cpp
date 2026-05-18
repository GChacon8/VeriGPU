// verigpu_backend.cpp — PyTorch custom backend for VeriGPU
// CP-3: allocator, device guard, copy, empty, fill, zero
// CP-4: add (Tensor, Scalar, in-place)
// CP-5: sub, mul, div, neg, abs, relu, clamp + in-place variants
// CP-6: sum, mean (full + dim), mm, addmm
// CP-7: autograd ops — view ops, threshold_backward, _local_scalar_dense
#include <ATen/detail/PrivateUse1HooksInterface.h>
#include <torch/extension.h>
#include <c10/core/impl/DeviceGuardImplInterface.h>
#include <cstring>
#include <cmath>
#include <algorithm>
#include <functional>
#include <vector>
#include <numeric>
#include <unordered_map>

// --- HW mode: Verilator runtime integration (CP-4 HW Roadmap) ---
// We forward-declare the runtime functions instead of #including
// gpu_runtime.h to avoid conflicts with its __global__/__device__ macros.
#ifdef VERIGPU_HW_AVAILABLE
extern void gpuCreateContext();
extern void* gpuMalloc(uint32_t requestedBytes);
extern void gpuCopyToDevice(void* gpuMemPtr, const void* srcData, size_t numBytes);
extern void gpuCopyFromDevice(void* destData, const void* gpuMemPtr, size_t numBytes);
extern void gpuLaunchKernel(const void* kernelPos, uint32_t numParams, const uint32_t* const p_params);
extern void gpuDestroyContext();
extern void gpuSetBaseThreadId(uint32_t base);
#endif
 
static bool g_hw_mode = false;

// Maps host pointers to GPU memory addresses
static std::unordered_map<void*, uint32_t> g_host_to_gpu;

static void sync_to_gpu(void* host_ptr, size_t nbytes) {
    if (!g_hw_mode || nbytes == 0) return;
    auto it = g_host_to_gpu.find(host_ptr);
    if (it != g_host_to_gpu.end()) {
        gpuCopyToDevice(reinterpret_cast<void*>(static_cast<size_t>(it->second)),
                        host_ptr, nbytes);
    }
}

static void sync_from_gpu(void* host_ptr, size_t nbytes) {
    if (!g_hw_mode || nbytes == 0) return;
    auto it = g_host_to_gpu.find(host_ptr);
    if (it != g_host_to_gpu.end()) {
        gpuCopyFromDevice(host_ptr,
                          reinterpret_cast<void*>(static_cast<size_t>(it->second)),
                          nbytes);
    }
}

static uint32_t get_gpu_addr(void* host_ptr) {
    auto it = g_host_to_gpu.find(host_ptr);
    return (it != g_host_to_gpu.end()) ? it->second : 0;
}

// ── Kernel infrastructure
static std::unordered_map<std::string, uint32_t> g_kernel_addrs;
static const uint32_t PARAM_BLOCK_ADDR = 0;

static void write_kernel_params(const std::vector<uint32_t>& params) {
    gpuCopyToDevice(
        reinterpret_cast<void*>(static_cast<size_t>(PARAM_BLOCK_ADDR)),
        params.data(),
        params.size() * sizeof(uint32_t));
}

static void launch_kernel(const std::string& name, uint32_t total_threads) {
    auto it = g_kernel_addrs.find(name);
    TORCH_CHECK(it != g_kernel_addrs.end(),
        "VeriGPU HW: kernel '", name, "' not loaded");
    uint32_t kernel_addr = it->second;
    for (uint32_t batch = 0; batch < total_threads; batch += 4) {
        gpuSetBaseThreadId(batch);
        gpuLaunchKernel(
            reinterpret_cast<void*>(static_cast<size_t>(kernel_addr)),
            0, nullptr);
    }
}   

namespace {

// =====================================================================
// 1. ALLOCATOR
// =====================================================================

static void verigpu_delete(void* ptr) {
    if (ptr) {
        // Remove from GPU address map
        g_host_to_gpu.erase(ptr);
        free(ptr);
    }
}
 
struct VeriGPUAllocator final : public c10::Allocator {
    at::DataPtr allocate(size_t nbytes) override {
        void* data = nullptr;
        if (nbytes > 0) {
            data = malloc(nbytes);
            TORCH_CHECK(data, "VeriGPU: alloc failed for ", nbytes, " bytes");
            memset(data, 0, nbytes);
 
            // In HW mode: also allocate on GPU and track the mapping
            if (g_hw_mode) {
                uint32_t gpu_addr = static_cast<uint32_t>(
                    reinterpret_cast<size_t>(gpuMalloc(static_cast<uint32_t>(nbytes))));
                g_host_to_gpu[data] = gpu_addr;
            }
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
 
    // Check if hardware mode is requested
    const char* hw_env = getenv("VERIGPU_USE_HW");
    if (hw_env && std::string(hw_env) == "1") {
#ifdef VERIGPU_HW_AVAILABLE
        g_hw_mode = true;
        gpuCreateContext();
        fprintf(stderr, "[VeriGPU] Hardware mode ENABLED (Verilator simulation active)\n");
#else
        fprintf(stderr, "[VeriGPU] WARNING: VERIGPU_USE_HW=1 but extension was built without HW support.\n");
        fprintf(stderr, "[VeriGPU]   Build libverigpu_runtime.so first, then rebuild the extension.\n");
        fprintf(stderr, "[VeriGPU]   Falling back to host-CPU mode.\n");
#endif
    }
 
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
// 2b. HOOKS INTERFACE (required by autograd in PyTorch 2.11+)
// =====================================================================

struct VeriGPUHooksInterface : public at::PrivateUse1HooksInterface {
    bool hasPrimaryContext(c10::DeviceIndex) const override {
        return true;
    }
};

static auto _hooks_reg = []() {
    at::RegisterPrivateUse1HooksInterface(new VeriGPUHooksInterface());
    return true;
}();

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

// ── Stride-aware copy helper (no PyTorch calls — no recursion) ──────
// Copies elements from src to dst respecting src's strides.
// dst is assumed contiguous. src may be non-contiguous (e.g. transposed).
static void strided_copy_to_contiguous(void* dst_ptr, const at::Tensor& src) {
    auto n = src.numel();
    auto dt = src.scalar_type();
    auto ndim = src.dim();
    auto sizes = src.sizes();
    auto strides = src.strides();

    // For each logical element, compute its offset via strides
    #define STRIDED_COPY(T) { \
        const T* sp = src.data_ptr<T>(); T* dp = static_cast<T*>(dst_ptr); \
        for (int64_t flat = 0; flat < n; flat++) { \
            int64_t offset = 0, idx = flat; \
            for (int64_t d = ndim - 1; d >= 0; d--) { \
                offset += (idx % sizes[d]) * strides[d]; \
                idx /= sizes[d]; \
            } \
            dp[flat] = sp[offset]; \
        } \
    }

    if      (dt == at::ScalarType::Float)  STRIDED_COPY(float)
    else if (dt == at::ScalarType::Double) STRIDED_COPY(double)
    else if (dt == at::ScalarType::Int)    STRIDED_COPY(int32_t)
    else if (dt == at::ScalarType::Long)   STRIDED_COPY(int64_t)
    else if (dt == at::ScalarType::Bool)   STRIDED_COPY(bool)
    else { TORCH_CHECK(false, "VeriGPU strided_copy: unsupported dtype"); }

    #undef STRIDED_COPY
}

at::Tensor& verigpu_copy_(at::Tensor& self, const at::Tensor& src, bool) {
    if (self.numel() == 0) return self;
    if (self.data_ptr() == src.data_ptr() && self.is_contiguous() && src.is_contiguous())
        return self;

    if (src.is_contiguous()) {
        auto sc = src;
        TORCH_CHECK(self.nbytes() == (size_t)sc.nbytes(), "VeriGPU copy_: size mismatch");
        memcpy(self.data_ptr(), sc.data_ptr(), self.nbytes());
    } else {
        // Non-contiguous source: direct stride-aware copy, no recursion
        strided_copy_to_contiguous(self.data_ptr(), src);
    }
    sync_to_gpu(self.data_ptr(), self.nbytes());
    return self;
}

at::Tensor verigpu_copy_from(const at::Tensor& self, const at::Tensor& dst, bool) {
    if (self.numel() == 0) return dst;

    if (self.is_contiguous()) {
        memcpy(dst.data_ptr(), self.data_ptr(), self.nbytes());
    } else {
        // Non-contiguous on our device: direct stride-aware copy to CPU dst
        strided_copy_to_contiguous(dst.data_ptr(), self);
    }
    return dst;
}

at::Tensor verigpu_copy_from_and_resize(const at::Tensor& self, const at::Tensor& dst) {
    dst.resize_as_(self);
    if (self.numel() == 0) return dst;

    if (self.is_contiguous()) {
        memcpy(dst.data_ptr(), self.data_ptr(), self.nbytes());
    } else {
        strided_copy_to_contiguous(dst.data_ptr(), self);
    }
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
    sync_to_gpu(self.data_ptr(), self.nbytes());
    return self;
}

at::Tensor& verigpu_zero_(at::Tensor& self) {
    if (self.nbytes() > 0) memset(self.data_ptr(), 0, self.nbytes());
    sync_to_gpu(self.data_ptr(), self.nbytes());
    return self;
}

// =====================================================================
// 5. ADD
// =====================================================================

at::Tensor verigpu_add_tensor(const at::Tensor& self, const at::Tensor& other, const at::Scalar& alpha) {
    auto [a, b, b_scalar] = prepare_binary(self, other);
    auto output = make_output_like(a); auto n = a.numel(); auto dtype = a.scalar_type();

    if (g_hw_mode && dtype == at::ScalarType::Float && alpha.toFloat() == 1.0f
        && !b_scalar && g_kernel_addrs.count("vadd_f32"))
    {
        sync_to_gpu(a.data_ptr(), a.nbytes());
        sync_to_gpu(b.data_ptr(), b.nbytes());

        uint32_t ga   = get_gpu_addr(a.data_ptr());
        uint32_t gb   = get_gpu_addr(b.data_ptr());
        uint32_t gout = get_gpu_addr(output.data_ptr());

        write_kernel_params({ga, gb, gout, static_cast<uint32_t>(n)});
        launch_kernel("vadd_f32", static_cast<uint32_t>(n));
        sync_from_gpu(output.data_ptr(), output.nbytes());

        return output;
    }

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

    if (g_hw_mode && dtype == at::ScalarType::Float && alpha.toFloat() == 1.0f
        && !b_scalar && g_kernel_addrs.count("vsub_f32"))
    {
        sync_to_gpu(a.data_ptr(), a.nbytes());
        sync_to_gpu(b.data_ptr(), b.nbytes());
        uint32_t ga   = get_gpu_addr(a.data_ptr());
        uint32_t gb   = get_gpu_addr(b.data_ptr());
        uint32_t gout = get_gpu_addr(output.data_ptr());
        write_kernel_params({ga, gb, gout, static_cast<uint32_t>(n)});
        launch_kernel("vsub_f32", static_cast<uint32_t>(n));
        sync_from_gpu(output.data_ptr(), output.nbytes());
        return output;
    }

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

at::Tensor verigpu_mul_tensor(const at::Tensor& s, const at::Tensor& o) {
    // HW MODE
    if (g_hw_mode && s.scalar_type() == at::ScalarType::Float
        && g_kernel_addrs.count("vmul_f32"))
    {
        auto a = s.contiguous();
        auto b = o.contiguous();
        if (b.dim() == 0 && b.scalar_type() != a.scalar_type()) b = b.to(a.scalar_type());
        if (b.dim() != 0 && a.sizes() == b.sizes()) {
            auto output = make_output_like(a);
            auto n = a.numel();
            sync_to_gpu(a.data_ptr(), a.nbytes());
            sync_to_gpu(b.data_ptr(), b.nbytes());
            uint32_t ga   = get_gpu_addr(a.data_ptr());
            uint32_t gb   = get_gpu_addr(b.data_ptr());
            uint32_t gout = get_gpu_addr(output.data_ptr());
            write_kernel_params({ga, gb, gout, static_cast<uint32_t>(n)});
            launch_kernel("vmul_f32", static_cast<uint32_t>(n));
            sync_from_gpu(output.data_ptr(), output.nbytes());
            return output;
        }
    }
    // HOST FALLBACK
    return binary_op(s, o, [](auto a, auto b){ return a*b; });
}


at::Tensor& verigpu_mul_tensor_(at::Tensor& self, const at::Tensor& other) {
    return binary_op_inplace(self, other, [](auto a, auto b){ return a*b; }); }


at::Tensor verigpu_mul_scalar(const at::Tensor& self, const at::Scalar& other) {
    return unary_op(self, [v=other.toDouble()](auto x){ return decltype(x)(x*v); }); }

// =====================================================================
// 8. DIV
// =====================================================================

at::Tensor verigpu_div_tensor(const at::Tensor& s, const at::Tensor& o) {
    // HW MODE
    if (g_hw_mode && s.scalar_type() == at::ScalarType::Float
        && g_kernel_addrs.count("vdiv_f32"))
    {
        auto a = s.contiguous();
        auto b = o.contiguous();
        if (b.dim() == 0 && b.scalar_type() != a.scalar_type()) b = b.to(a.scalar_type());
        if (b.dim() != 0 && a.sizes() == b.sizes()) {
            auto output = make_output_like(a);
            auto n = a.numel();
            uint32_t ga   = get_gpu_addr(a.data_ptr());
            uint32_t gb   = get_gpu_addr(b.data_ptr());
            uint32_t gout = get_gpu_addr(output.data_ptr());
            write_kernel_params({ga, gb, gout, static_cast<uint32_t>(n)});
            launch_kernel("vdiv_f32", static_cast<uint32_t>(n));
            sync_from_gpu(output.data_ptr(), output.nbytes());
            return output;
        }
    }
    // HOST FALLBACK
    return binary_op(s, o, [](auto a, auto b){ return a/b; });
}

    
at::Tensor& verigpu_div_tensor_(at::Tensor& self, const at::Tensor& other) {
    return binary_op_inplace(self, other, [](auto a, auto b){ return a/b; }); }
at::Tensor verigpu_div_scalar(const at::Tensor& self, const at::Scalar& other) {
    return unary_op(self, [v=other.toDouble()](auto x){ return decltype(x)(x/v); }); }

// =====================================================================
// 9. UNARY OPS
// =====================================================================

at::Tensor verigpu_neg(const at::Tensor& s) {
    // HW MODE (unary: 3 params)
    if (g_hw_mode && s.scalar_type() == at::ScalarType::Float
        && g_kernel_addrs.count("vneg_f32"))
    {
        auto a = s.contiguous();
        auto output = make_output_like(a);
        auto n = a.numel();
        sync_to_gpu(a.data_ptr(), a.nbytes());
        uint32_t ga   = get_gpu_addr(a.data_ptr());
        uint32_t gout = get_gpu_addr(output.data_ptr());
        write_kernel_params({ga, gout, static_cast<uint32_t>(n)});
        launch_kernel("vneg_f32", static_cast<uint32_t>(n));
        sync_from_gpu(output.data_ptr(), output.nbytes());
        return output;
    }
    // HOST FALLBACK
    return unary_op(s, [](auto x){ return -x; });
}


at::Tensor verigpu_abs(const at::Tensor& s) {
    // HW MODE (unary: 3 params)
    if (g_hw_mode && s.scalar_type() == at::ScalarType::Float
        && g_kernel_addrs.count("vabs_f32"))
    {
        auto a = s.contiguous();
        auto output = make_output_like(a);
        auto n = a.numel();
        sync_to_gpu(a.data_ptr(), a.nbytes());
        uint32_t ga   = get_gpu_addr(a.data_ptr());
        uint32_t gout = get_gpu_addr(output.data_ptr());
        write_kernel_params({ga, gout, static_cast<uint32_t>(n)});
        launch_kernel("vabs_f32", static_cast<uint32_t>(n));
        sync_from_gpu(output.data_ptr(), output.nbytes());
        return output;
    }
    // HOST FALLBACK
    return unary_op(s, [](auto x){ return x<0?-x:x; });
}


at::Tensor verigpu_relu(const at::Tensor& s) {
    // HW MODE (unary: 3 params)
    if (g_hw_mode && s.scalar_type() == at::ScalarType::Float
        && g_kernel_addrs.count("vrelu_f32"))
    {
        auto a = s.contiguous();
        auto output = make_output_like(a);
        auto n = a.numel();
        sync_to_gpu(a.data_ptr(), a.nbytes());
        uint32_t ga   = get_gpu_addr(a.data_ptr());
        uint32_t gout = get_gpu_addr(output.data_ptr());
        write_kernel_params({ga, gout, static_cast<uint32_t>(n)});
        launch_kernel("vrelu_f32", static_cast<uint32_t>(n));
        sync_from_gpu(output.data_ptr(), output.nbytes());
        return output;
    }
    // HOST FALLBACK
    return unary_op(s, [](auto x){ return x>0?x:decltype(x)(0); });
}


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

    if (g_hw_mode && a.scalar_type() == at::ScalarType::Float
        && g_kernel_addrs.count("vsum_f32"))
    {
        sync_to_gpu(a.data_ptr(), a.nbytes());
 
        // Create scalar output tensor on device
        auto output = torch::zeros({}, at::TensorOptions()
            .dtype(at::ScalarType::Float).device(a.device()));
 
        uint32_t ga   = get_gpu_addr(a.data_ptr());
        uint32_t gout = get_gpu_addr(output.data_ptr());
 
        write_kernel_params({ga, gout, static_cast<uint32_t>(a.numel())});
        launch_kernel("vsum_f32", 4);  // REVISAR ESTO, NO ESTOY SEGUROOOOOOOOOOOOOOOOOO
        sync_from_gpu(output.data_ptr(), output.nbytes());
 
        return output;
    }

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

    if (g_hw_mode && a.scalar_type() == at::ScalarType::Float
        && g_kernel_addrs.count("vmm_f32"))
    {
        sync_to_gpu(a.data_ptr(), a.nbytes());
        sync_to_gpu(b.data_ptr(), b.nbytes());
 
        uint32_t ga   = get_gpu_addr(a.data_ptr());
        uint32_t gb   = get_gpu_addr(b.data_ptr());
        uint32_t gc   = get_gpu_addr(output.data_ptr());
 
        // 6 parameters: addr_A, addr_B, addr_C, M, K, N
        write_kernel_params({ga, gb, gc,
            static_cast<uint32_t>(M),
            static_cast<uint32_t>(K),
            static_cast<uint32_t>(N)});
 
        // Each thread computes one row of C. With 4 cores,
        // M rows need ceil(M/4) batches.
        launch_kernel("vmm_f32", static_cast<uint32_t>(M));
 
        sync_from_gpu(output.data_ptr(), output.nbytes());
        return output;
    }

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
// 15. VIA-CPU HELPER (CP-7)
// =====================================================================
// Run an op on CPU and bring result back to our device.
// Since our GPU memory IS host memory, this is just memcpy overhead.

static at::Tensor via_cpu(const at::Tensor& self,
    std::function<at::Tensor(const at::Tensor&)> cpu_op)
{
    auto cpu_input = self.cpu();
    auto cpu_result = cpu_op(cpu_input).contiguous();
    auto out = make_verigpu_contiguous(cpu_result.sizes(), cpu_result.scalar_type());
    memcpy(out.data_ptr(), cpu_result.data_ptr(), cpu_result.nbytes());
    return out;
}

// =====================================================================
// 16. VIEW OPS (CP-7) — shared-storage operations for autograd
// =====================================================================
// These create new tensor views or copies that autograd's backward
// formulas need internally: transpose for mm backward, reshape for
// gradient shape manipulation, expand for sum backward, etc.

// as_strided: fundamental view op — shares storage, changes sizes/strides
at::Tensor verigpu_as_strided(const at::Tensor& self, at::IntArrayRef size,
    at::IntArrayRef stride, std::optional<int64_t> storage_offset)
{
    auto offset = storage_offset.value_or(self.storage_offset());
    auto tensor = at::detail::make_tensor<c10::TensorImpl>(
        c10::Storage(self.storage()),
        c10::DispatchKeySet(c10::DispatchKey::PrivateUse1),
        self.dtype());
    tensor.unsafeGetTensorImpl()->set_sizes_and_strides(size, stride);
    tensor.unsafeGetTensorImpl()->set_storage_offset(offset);
    return tensor;
}

at::Tensor verigpu_t(const at::Tensor& self) {
    TORCH_CHECK(self.dim() <= 2, "VeriGPU t: need ≤2D, got ", self.dim());
    if (self.dim() < 2) return self;
    return via_cpu(self, [](const at::Tensor& cpu) {
        return cpu.t().contiguous();
    });
}

at::Tensor verigpu_transpose(const at::Tensor& self, int64_t dim0, int64_t dim1) {
    return via_cpu(self, [&](const at::Tensor& cpu) {
        return cpu.transpose(dim0, dim1).contiguous();
    });
}

// view / reshape
at::Tensor verigpu_view(const at::Tensor& self, at::IntArrayRef shape) {
    return via_cpu(self, [&](const at::Tensor& cpu) {
        return cpu.view(shape).contiguous();
    });
}

at::Tensor verigpu_reshape(const at::Tensor& self, at::IntArrayRef shape) {
    return via_cpu(self, [&](const at::Tensor& cpu) {
        return cpu.reshape(shape).contiguous();
    });
}

// unsqueeze — add a dimension of size 1
at::Tensor verigpu_unsqueeze(const at::Tensor& self, int64_t dim) {
    auto ndim = self.dim();
    if (dim < 0) dim += ndim + 1;
    auto sizes = self.sizes().vec();
    auto strides = self.strides().vec();
    int64_t new_stride = (dim < ndim && !strides.empty()) ? strides[dim] : 1;
    if (dim > 0 && dim <= (int64_t)strides.size()) {
        new_stride = sizes[dim - 1] * strides[dim - 1];
    }
    sizes.insert(sizes.begin() + dim, 1);
    strides.insert(strides.begin() + dim, new_stride);
    return verigpu_as_strided(self, sizes, strides, self.storage_offset());
}

// expand — broadcast a tensor to a larger size
at::Tensor verigpu_expand(const at::Tensor& self, at::IntArrayRef size, bool) {
    return via_cpu(self, [&](const at::Tensor& cpu) {
        return cpu.expand(size).contiguous();
    });
}

// slice — extract a sub-range along a dimension
at::Tensor verigpu_slice(const at::Tensor& self, int64_t dim,
    std::optional<int64_t> start, std::optional<int64_t> end, int64_t step)
{
    return via_cpu(self, [&](const at::Tensor& cpu) {
        return cpu.slice(dim, start, end, step).contiguous();
    });
}

// =====================================================================
// 17. GRADIENT-SPECIFIC OPS (CP-7)
// =====================================================================

// threshold_backward: gradient of relu
// grad_input = grad_output * (self > threshold)
at::Tensor verigpu_threshold_backward(const at::Tensor& grad_output,
    const at::Tensor& self, const at::Scalar& threshold)
{
    auto g = grad_output.contiguous();
    auto s = self.contiguous();
    auto output = make_output_like(g);
    auto n = g.numel();
    auto dtype = g.scalar_type();

    if (g_hw_mode && grad_output.scalar_type() == at::ScalarType::Float
        && threshold.toFloat() == 0.0f
        && g_kernel_addrs.count("vthreshold_bwd_f32"))
    {
        auto grad = grad_output.contiguous();
        auto inp  = self.contiguous();
        auto output = make_output_like(grad);
        auto n = grad.numel();

        sync_to_gpu(grad.data_ptr(), grad.nbytes());
        sync_to_gpu(inp.data_ptr(), inp.nbytes());

        uint32_t g_grad = get_gpu_addr(grad.data_ptr());
        uint32_t g_inp  = get_gpu_addr(inp.data_ptr());
        uint32_t g_out  = get_gpu_addr(output.data_ptr());
        write_kernel_params({g_grad, g_inp, g_out, static_cast<uint32_t>(n)});
        launch_kernel("vthreshold_bwd_f32", static_cast<uint32_t>(n));
        sync_from_gpu(output.data_ptr(), output.nbytes());
        return output;
    }

    if (dtype == at::ScalarType::Float) {
        const float* pg = g.data_ptr<float>();
        const float* ps = s.data_ptr<float>();
        float* po = output.data_ptr<float>();
        float thresh = threshold.toFloat();
        for (int64_t i = 0; i < n; i++)
            po[i] = ps[i] > thresh ? pg[i] : 0.0f;
    } else if (dtype == at::ScalarType::Double) {
        const double* pg = g.data_ptr<double>();
        const double* ps = s.data_ptr<double>();
        double* po = output.data_ptr<double>();
        double thresh = threshold.toDouble();
        for (int64_t i = 0; i < n; i++)
            po[i] = ps[i] > thresh ? pg[i] : 0.0;
    } else {
        TORCH_CHECK(false, "VeriGPU threshold_backward: float/double only");
    }
    return output;
}

// _local_scalar_dense: extract a Python scalar from a 0-dim or 1-element tensor.
// Called by loss.item(), and by some autograd internals.
at::Scalar verigpu_local_scalar_dense(const at::Tensor& self) {
    TORCH_CHECK(self.numel() == 1, "VeriGPU _local_scalar_dense: need 1 element");
    auto a = self.contiguous();
    auto dt = a.scalar_type();
    if (dt == at::ScalarType::Float) return at::Scalar(*a.data_ptr<float>());
    if (dt == at::ScalarType::Double) return at::Scalar(*a.data_ptr<double>());
    if (dt == at::ScalarType::Int) return at::Scalar(*a.data_ptr<int32_t>());
    if (dt == at::ScalarType::Long) return at::Scalar(*a.data_ptr<int64_t>());
    if (dt == at::ScalarType::Bool) return at::Scalar(*a.data_ptr<bool>());
    TORCH_CHECK(false, "VeriGPU _local_scalar_dense: unsupported dtype");
}


// =====================================================================
// 18. REGISTER ALL OPERATIONS
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

    // CP-7: view ops (for autograd backward)
    m.impl("as_strided",             &verigpu_as_strided);
    m.impl("t",                      &verigpu_t);
    m.impl("transpose.int",          &verigpu_transpose);
    m.impl("view",                   &verigpu_view);
    m.impl("reshape",                &verigpu_reshape);
    m.impl("unsqueeze",              &verigpu_unsqueeze);
    m.impl("expand",                 &verigpu_expand);
    m.impl("slice.Tensor",           &verigpu_slice);

    // CP-7: gradient ops
    m.impl("threshold_backward",     &verigpu_threshold_backward);
    m.impl("_local_scalar_dense",    &verigpu_local_scalar_dense);
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
    m.def("is_hw_mode", []() -> bool { return g_hw_mode; });
    m.def("hw_available", []() -> bool {
#ifdef VERIGPU_HW_AVAILABLE
        return true;
#else
        return false;
#endif
    });

    m.def("gpu_addr_of", [](const at::Tensor& t) -> int64_t {
        if (!g_hw_mode) return -1;
        auto it = g_host_to_gpu.find(t.data_ptr());
        if (it != g_host_to_gpu.end()) return static_cast<int64_t>(it->second);
        return -1;
    });
 
    m.def("gpu_readback_floats", [](int64_t gpu_addr, int64_t count) -> std::vector<float> {
        std::vector<float> result(count);
        if (g_hw_mode && gpu_addr >= 0) {
            gpuCopyFromDevice(result.data(),
                              reinterpret_cast<void*>(static_cast<size_t>(gpu_addr)),
                              count * sizeof(float));
        }
        return result;
    });

    m.def("load_kernel", [](const std::string& name, std::vector<uint32_t> words) {
        if (!g_hw_mode)
            throw std::runtime_error("Cannot load kernel: HW mode not active");
        uint32_t size_bytes = words.size() * sizeof(uint32_t);
        uint32_t addr = static_cast<uint32_t>(
            reinterpret_cast<size_t>(gpuMalloc(size_bytes)));
        gpuCopyToDevice(
            reinterpret_cast<void*>(static_cast<size_t>(addr)),
            words.data(), size_bytes);
        g_kernel_addrs[name] = addr;
        fprintf(stderr, "[VeriGPU] Loaded kernel '%s' (%zu instructions) at GPU addr %u\n",
                name.c_str(), words.size(), addr);
    });

    m.def("list_kernels", []() -> std::vector<std::string> {
        std::vector<std::string> names;
        for (auto& kv : g_kernel_addrs) names.push_back(kv.first);
        return names;
    });
}
