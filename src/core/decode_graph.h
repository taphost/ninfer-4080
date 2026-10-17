#pragma once

#include <cuda_runtime.h>

#include <functional>

namespace ninfer {

class DecodeGraphDefinition {
public:
    DecodeGraphDefinition() = default;
    ~DecodeGraphDefinition();

    DecodeGraphDefinition(const DecodeGraphDefinition&)            = delete;
    DecodeGraphDefinition& operator=(const DecodeGraphDefinition&) = delete;
    DecodeGraphDefinition(DecodeGraphDefinition&& other) noexcept;
    DecodeGraphDefinition& operator=(DecodeGraphDefinition&& other) noexcept;

    void capture(cudaStream_t stream, const std::function<void()>& body);
    [[nodiscard]] bool ready() const noexcept;
    void reset() noexcept;

private:
    friend class DecodeGraphExecutable;
    cudaGraph_t graph_ = nullptr;
};

class DecodeGraphExecutable {
public:
    DecodeGraphExecutable() = default;
    ~DecodeGraphExecutable();

    DecodeGraphExecutable(const DecodeGraphExecutable&)            = delete;
    DecodeGraphExecutable& operator=(const DecodeGraphExecutable&) = delete;
    DecodeGraphExecutable(DecodeGraphExecutable&& other) noexcept;
    DecodeGraphExecutable& operator=(DecodeGraphExecutable&& other) noexcept;

    void instantiate(const DecodeGraphDefinition& definition);
    void update(const DecodeGraphDefinition& definition);
    // In-place update with fallback: tries cudaGraphExecUpdate, and on failure
    // re-instantiates the executable from the new definition instead of throwing.
    // Fallback is on by default; set_graph_update_fallback(false) restores update()'s behavior.
    void update_or_reinstantiate(const DecodeGraphDefinition& definition);
    void upload(cudaStream_t stream);
    void launch(cudaStream_t stream);
    [[nodiscard]] bool ready() const noexcept;
    void reset() noexcept;

    // Runtime policy: when a graph update fails, re-instantiate instead of failing.
    // Default true. Off keeps update()/update_or_reinstantiate() throwing.
    static void set_graph_update_fallback(bool enabled) noexcept;
    [[nodiscard]] static bool graph_update_fallback() noexcept;

private:
    cudaGraphExec_t exec_ = nullptr;
};

} // namespace ninfer
