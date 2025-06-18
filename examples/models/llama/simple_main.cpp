/*
 * Simple main.cpp for llama_main without gflags dependency
 */

#include <executorch/examples/models/llama/runner/runner.h>
#include <executorch/runtime/platform/runtime.h>

#if defined(ET_USE_THREADPOOL)
#include <executorch/extension/threadpool/cpuinfo_utils.h>
#include <executorch/extension/threadpool/threadpool.h>
#endif

#include <iostream>
#include <memory>

int main(int argc, char* argv[]) {
    if (argc < 3) {
        std::cerr << "Usage: " << argv[0] << " --model_path <model.pte> --tokenizer_path <tokenizer.model> [--prompt <prompt>] [--cpu_threads <num>] [--warmup]" << std::endl;
        return 1;
    }

    // Parse simple command line arguments
    std::string model_path;
    std::string tokenizer_path;
    std::string prompt = "The answer to the ultimate question is";
    int cpu_threads = 5;
    bool warmup = false;
    float temperature = 0.8f;
    int seq_len = 128;

    for (int i = 1; i < argc; i++) {
        if (std::string(argv[i]) == "--model_path" && i + 1 < argc) {
            model_path = argv[++i];
        } else if (std::string(argv[i]) == "--tokenizer_path" && i + 1 < argc) {
            tokenizer_path = argv[++i];
        } else if (std::string(argv[i]) == "--prompt" && i + 1 < argc) {
            prompt = argv[++i];
        } else if (std::string(argv[i]) == "--cpu_threads" && i + 1 < argc) {
            cpu_threads = std::stoi(argv[++i]);
        } else if (std::string(argv[i]) == "--warmup") {
            warmup = true;
        }
    }

    if (model_path.empty() || tokenizer_path.empty()) {
        std::cerr << "Both --model_path and --tokenizer_path are required" << std::endl;
        return 1;
    }

    // Initialize ExecuTorch runtime
    executorch::runtime::runtime_init();

#if defined(ET_USE_THREADPOOL)
    uint32_t num_performant_cores = cpu_threads == -1
        ? ::executorch::extension::cpuinfo::get_num_performant_cores()
        : static_cast<uint32_t>(cpu_threads);
    ET_LOG(
        Info, "Resetting threadpool with num threads = %d", num_performant_cores);
    if (num_performant_cores > 0) {
        ::executorch::extension::threadpool::get_threadpool()
            ->_unsafe_reset_threadpool(num_performant_cores);
    }
#endif

    // Create llama runner
    std::unique_ptr<::executorch::extension::llm::TextLLMRunner> runner =
        example::create_llama_runner(model_path, tokenizer_path, std::nullopt, temperature);

    if (runner == nullptr) {
        ET_LOG(Error, "Failed to create llama runner");
        return 1;
    }

    if (warmup) {
        ET_LOG(Info, "Doing a warmup run...");
        runner->warmup(prompt, /*max_new_tokens=*/seq_len);
        ET_LOG(Info, "Warmup run finished!");
    }

    // Generate
    executorch::extension::llm::GenerationConfig config{
        .seq_len = seq_len, .temperature = temperature};
    
    ET_LOG(Info, "Starting generation...");
    std::cout << prompt;
    runner->generate(prompt, config);

    return 0;
}
