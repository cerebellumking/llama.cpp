#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <limits>
#include <utility>
#include <vector>

#if defined(__linux__)
#include <sched.h>
#endif

namespace heterospec_hotpath {

inline void append_u32_le(std::vector<uint8_t>& out, uint32_t value) {
    out.push_back(static_cast<uint8_t>(value & 0xFFu));
    out.push_back(static_cast<uint8_t>((value >> 8u) & 0xFFu));
    out.push_back(static_cast<uint8_t>((value >> 16u) & 0xFFu));
    out.push_back(static_cast<uint8_t>((value >> 24u) & 0xFFu));
}

inline void append_i32_le(std::vector<uint8_t>& out, int32_t value) {
    append_u32_le(out, static_cast<uint32_t>(value));
}

inline std::vector<uint8_t> pack_token_request_v3(
        uint8_t type,
        uint32_t request_id,
        const std::vector<int32_t>& tokens) {
    std::vector<uint8_t> out;
    out.reserve(1 + 2 * sizeof(uint32_t) + tokens.size() * sizeof(int32_t));
    out.push_back(type);
    append_u32_le(out, request_id);
    append_u32_le(out, static_cast<uint32_t>(tokens.size()));
    for (int32_t token : tokens) {
        append_i32_le(out, token);
    }
    return out;
}

inline bool greedy_argmax(
        const float * logits,
        int n_vocab,
        int32_t& token,
        float& confidence) {
    if (logits == nullptr || n_vocab <= 0) {
        return false;
    }
    int best_id = -1;
    float best_logit = -std::numeric_limits<float>::infinity();
    float second_best_logit = -std::numeric_limits<float>::infinity();
    for (int i = 0; i < n_vocab; ++i) {
        const float value = logits[i];
        if (!std::isfinite(value)) {
            continue;
        }
        if (value > best_logit) {
            second_best_logit = best_logit;
            best_logit = value;
            best_id = i;
        } else if (value > second_best_logit) {
            second_best_logit = value;
        }
    }
    if (best_id < 0) {
        return false;
    }
    token = static_cast<int32_t>(best_id);
    const double margin = std::isfinite(second_best_logit)
            ? static_cast<double>(best_logit - second_best_logit)
            : 0.0;
    confidence = static_cast<float>(1.0 / (1.0 + std::exp(-margin)));
    confidence = std::max(0.0f, std::min(1.0f, confidence));
    return true;
}

inline int decode_thread_count(int nproc) {
    if (nproc <= 1) {
        return 1;
    }
    if (nproc >= 8) {
        return 6;
    }
    return std::max(1, std::min(6, nproc - 2));
}

inline int cpu_capacity(int cpu_index) {
    int cap = 1024;
#if defined(__linux__)
    char path[96];
    std::snprintf(path, sizeof(path), "/sys/devices/system/cpu/cpu%d/cpu_capacity", cpu_index);
    FILE * file = std::fopen(path, "r");
    if (file != nullptr) {
        if (std::fscanf(file, "%d", &cap) != 1) {
            cap = 1024;
        }
        std::fclose(file);
    }
#else
    (void)cpu_index;
#endif
    return cap;
}

inline std::vector<int> big_core_ids(int nproc, int n_threads) {
    const int total = std::max(1, nproc);
    const int want = std::max(1, std::min(n_threads, total));
    std::vector<std::pair<int, int>> ranked;
    ranked.reserve(static_cast<size_t>(total));
    for (int index = 0; index < total; ++index) {
        ranked.emplace_back(cpu_capacity(index), index);
    }
    std::sort(ranked.begin(), ranked.end(), [](const std::pair<int, int>& left, const std::pair<int, int>& right) {
        if (left.first != right.first) {
            return left.first > right.first;
        }
        return left.second < right.second;
    });
    std::vector<int> ids;
    ids.reserve(static_cast<size_t>(want));
    for (int index = 0; index < want; ++index) {
        ids.push_back(ranked[static_cast<size_t>(index)].second);
    }
    std::sort(ids.begin(), ids.end());
    return ids;
}

inline bool pin_to_cpus(const std::vector<int>& ids) {
#if defined(__linux__)
    if (ids.empty()) {
        return false;
    }
    cpu_set_t set;
    CPU_ZERO(&set);
    for (int id : ids) {
        if (id >= 0) {
            CPU_SET(id, &set);
        }
    }
    return sched_setaffinity(0, sizeof(set), &set) == 0;
#else
    (void)ids;
    return false;
#endif
}

inline bool overlap_hit(
        int accepted,
        int sent_k,
        int32_t bonus,
        int32_t guessed_bonus,
        int prefetch_k) {
    return sent_k > 0
            && accepted >= sent_k
            && bonus == guessed_bonus
            && prefetch_k > 0;
}

// Generate leaves the last sent draft out of KV. Overlap writes it at pos_base+k,
// then the guessed bonus at pos_base+k+1. On miss, drop everything after the
// last accepted draft: [pos_base + accepted + 1, inf).
inline int overlap_kv_trim_pos(int pos_base, int accepted) {
    if (pos_base < 0) {
        pos_base = 0;
    }
    if (accepted < 0) {
        accepted = 0;
    }
    return pos_base + accepted + 1;
}

inline int overlap_first_extra_pos(int pos_base, int sent_k) {
    return pos_base + std::max(0, sent_k);
}

}  // namespace heterospec_hotpath
