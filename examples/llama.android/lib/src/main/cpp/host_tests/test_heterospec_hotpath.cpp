#include "heterospec_hotpath.hpp"

#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <vector>

static int g_failures = 0;

static void expect(bool cond, const char * msg) {
    if (!cond) {
        std::fprintf(stderr, "FAIL %s\n", msg);
        ++g_failures;
    }
}

int main() {
    float logits[8] = {0.1f, 3.0f, 2.5f, -1.0f, 0.0f, 1.2f, 0.4f, 0.2f};
    int32_t token = -1;
    float confidence = 0.0f;
    expect(heterospec_hotpath::greedy_argmax(logits, 8, token, confidence), "argmax ok");
    expect(token == 1, "argmax token");
    expect(confidence > 0.5f && confidence < 1.0f, "confidence in (0.5, 1)");

    const auto packed = heterospec_hotpath::pack_token_request_v3(0x11, 7, {10, 11, 12});
    expect(packed.size() == 1 + 8 + 12, "pack size");
    expect(packed[0] == 0x11, "msg type");
    expect(packed[1] == 7 && packed[2] == 0 && packed[3] == 0 && packed[4] == 0, "request id le");
    expect(packed[5] == 3, "token count");

    expect(heterospec_hotpath::decode_thread_count(8) == 6, "8 gen3 uses 6 threads");
    expect(heterospec_hotpath::decode_thread_count(4) == 2, "4 cores leave 2");
    expect(heterospec_hotpath::decode_thread_count(1) == 1, "single core");

    const auto cores = heterospec_hotpath::big_core_ids(8, 6);
    expect(cores.size() == 6, "six big cores");
    expect(cores.front() >= 0 && cores.back() < 8, "core ids in range");
    expect(heterospec_hotpath::overlap_hit(3, 3, 9, 9, 3), "full accept bonus match");
    expect(!heterospec_hotpath::overlap_hit(2, 3, 9, 9, 3), "partial accept is miss");
    expect(!heterospec_hotpath::overlap_hit(3, 3, 8, 9, 3), "wrong bonus is miss");
    expect(!heterospec_hotpath::overlap_hit(3, 3, 9, 9, 0), "empty prefetch is miss");
    expect(heterospec_hotpath::overlap_kv_trim_pos(10, 3) == 14, "full accept drops guessed bonus");
    expect(heterospec_hotpath::overlap_kv_trim_pos(10, 0) == 11, "zero accept keeps last token only");
    expect(heterospec_hotpath::overlap_kv_trim_pos(10, 2) == 13, "partial accept drops rejected tail");
    expect(heterospec_hotpath::overlap_first_extra_pos(10, 3) == 13, "last sent draft is first extra");

    // Commit position: [previous last] + accepted drafts, starting at the old prompt length.
    // Phone stall log: prompt=32, accepted=1 -> two tokens at positions 32 and 33.
    {
        const int prompt_len = 32;
        expect(heterospec_hotpath::commit_pos0(prompt_len) == 32, "accept 0 pos0");
        expect(heterospec_hotpath::commit_token_count(0) == 1, "accept 0 commits previous last only");
        expect(!heterospec_hotpath::keep_draft_branch(false), "accept 0 drops the draft branch");
    }
    {
        const int accepted = 1;
        expect(heterospec_hotpath::commit_pos0(32) == 32, "partial accept pos0");
        expect(heterospec_hotpath::commit_token_count(accepted) == 2, "partial accept commits last plus one draft");
        expect(!heterospec_hotpath::keep_draft_branch(
                heterospec_hotpath::overlap_hit(accepted, 3, 198, 271, 2)),
                "partial accept drops the draft branch");
    }
    {
        const int accepted = 3;
        expect(heterospec_hotpath::commit_pos0(32) == 32, "full accept pos0");
        expect(heterospec_hotpath::commit_token_count(accepted) == 4, "full accept commits last plus every draft");
        expect(!heterospec_hotpath::keep_draft_branch(
                heterospec_hotpath::overlap_hit(accepted, 3, 8, 9, 2)),
                "full accept with wrong bonus drops the draft branch");
    }
    {
        const int accepted = 3;
        const bool hit = heterospec_hotpath::overlap_hit(accepted, 3, 9, 9, 2);
        expect(hit, "hit accepts every sent draft and the guessed bonus");
        expect(heterospec_hotpath::commit_pos0(32) == 32, "hit pos0");
        expect(heterospec_hotpath::commit_token_count(accepted) == 4, "hit still commits the accepted prefix");
        expect(heterospec_hotpath::keep_draft_branch(hit), "hit keeps the draft branch");
    }
    expect(heterospec_hotpath::commit_token_count(-1) == 1, "negative accept clamps");
    expect(heterospec_hotpath::commit_pos0(-4) == 0, "negative prompt clamps");
    expect(heterospec_hotpath::prompt_token_budget(3072, 512) == 2560, "leave 512 tokens to generate");
    expect(heterospec_hotpath::prompt_token_budget(512, 512) == 1, "predict filling the context keeps one prompt token");
    expect(heterospec_hotpath::prefill_chunk_size(512, 1536) == 512, "hybrid prefill chunk is n_batch");
    expect(heterospec_hotpath::prefill_chunk_size(4096, 1536) == 1536, "prefill chunk stays inside the allocated batch");
    expect(heterospec_hotpath::prefetch_leaves_kv_hole(true, 3, 3), "full prefetch accept leaves the last draft out of KV");
    expect(!heterospec_hotpath::prefetch_leaves_kv_hole(true, 2, 3), "partial prefetch accept is already in KV");
    expect(!heterospec_hotpath::prefetch_leaves_kv_hole(false, 3, 3), "overlap writes the last sent draft");
    // Last-year ACK: n_past starts at pos_base-1, then n_past += accepted+1,
    // then seq_rm(n_past+1, -1). That start index is overlap_kv_trim_pos.
    for (int pos_base = 1; pos_base <= 12; ++pos_base) {
        for (int accepted = 0; accepted <= 4; ++accepted) {
            const int n_past_start = pos_base - 1;
            const int n_past_after = n_past_start + accepted + 1;
            const int legacy_seq_rm = n_past_after + 1;
            expect(
                    heterospec_hotpath::overlap_kv_trim_pos(pos_base, accepted) == legacy_seq_rm,
                    "overlap miss trim equals 2025 seq_rm(n_past+1)");
        }
    }

    if (g_failures != 0) {
        std::fprintf(stderr, "%d failures\n", g_failures);
        return 1;
    }
    std::puts("heterospec_hotpath host tests passed");
    return 0;
}
