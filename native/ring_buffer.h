#ifndef PYSYSAUDIO_RING_BUFFER_H
#define PYSYSAUDIO_RING_BUFFER_H

#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstddef>
#include <cstring>
#include <mutex>
#include <vector>

namespace pysysaudio {

// Single-producer, single-consumer queue of interleaved float32 samples.
//
// The producer is the platform's audio capture thread. push() never blocks,
// never allocates and never takes a lock, so it is safe to call from a
// real-time context. When there is not enough room the incoming chunk is
// dropped whole, which keeps the stored data frame-aligned, and the drop is
// counted so the consumer can report it.
//
// The consumer is the Python reader thread. wait_and_pop() blocks (with the
// GIL released, see bindings.cpp) until samples arrive, the queue is closed,
// or the timeout passes. Only one consumer may read at a time.
class SampleQueue {
public:
    // Discard contents and size the queue for `capacity` samples.
    // Call only while no producer is running (before capture starts).
    void reset(size_t capacity) {
        std::lock_guard<std::mutex> lock(mutex_);
        data_.assign(std::max<size_t>(capacity, 1), 0.0f);
        capacity_ = data_.size();
        write_pos_.store(0, std::memory_order_relaxed);
        read_pos_.store(0, std::memory_order_relaxed);
        dropped_.store(0, std::memory_order_relaxed);
        closed_ = false;
    }

    // Producer side. Returns false if the chunk did not fit and was dropped.
    bool push(const float* samples, size_t count) noexcept {
        if (count == 0) {
            return true;
        }
        const size_t w = write_pos_.load(std::memory_order_relaxed);
        const size_t r = read_pos_.load(std::memory_order_acquire);
        if (count > capacity_ - (w - r)) {
            dropped_.fetch_add(count, std::memory_order_relaxed);
            return false;
        }

        const size_t start = w % capacity_;
        const size_t first = std::min(count, capacity_ - start);
        std::memcpy(data_.data() + start, samples, first * sizeof(float));
        if (first < count) {
            std::memcpy(data_.data(), samples + first, (count - first) * sizeof(float));
        }
        write_pos_.store(w + count, std::memory_order_release);

        // Notifying without holding mutex_ keeps the producer lock-free. A wakeup
        // can be missed as a result; the consumer bounds that by waiting in slices.
        cv_.notify_one();
        return true;
    }

    // Consumer side. Replaces the contents of `out` with up to max_count samples.
    // Returns once at least one sample was copied, the queue is closed, or
    // timeout_seconds elapsed. A negative timeout waits indefinitely.
    void wait_and_pop(std::vector<float>& out, size_t max_count, double timeout_seconds) {
        out.clear();
        if (max_count == 0) {
            return;
        }

        std::unique_lock<std::mutex> lock(mutex_);
        const bool forever = timeout_seconds < 0.0;
        const auto deadline =
            std::chrono::steady_clock::now() +
            std::chrono::duration_cast<std::chrono::steady_clock::duration>(
                std::chrono::duration<double>(forever ? 0.0 : timeout_seconds));
        constexpr auto slice = std::chrono::milliseconds(10);

        while (available() == 0 && !closed_) {
            const auto now = std::chrono::steady_clock::now();
            if (!forever && now >= deadline) {
                return;
            }
            auto until = now + slice;
            if (!forever && deadline < until) {
                until = deadline;
            }
            cv_.wait_until(lock, until);
        }
        pop_locked(out, max_count);
    }

    // Wake any waiting consumer; subsequent reads drain what is left and then
    // return empty.
    void close() {
        {
            std::lock_guard<std::mutex> lock(mutex_);
            closed_ = true;
        }
        cv_.notify_all();
    }

    size_t available() const noexcept {
        return write_pos_.load(std::memory_order_acquire) -
               read_pos_.load(std::memory_order_acquire);
    }

    size_t dropped() const noexcept { return dropped_.load(std::memory_order_relaxed); }

private:
    void pop_locked(std::vector<float>& out, size_t max_count) {
        const size_t r = read_pos_.load(std::memory_order_relaxed);
        const size_t w = write_pos_.load(std::memory_order_acquire);
        const size_t count = std::min(w - r, max_count);
        if (count == 0) {
            return;
        }

        out.resize(count);
        const size_t start = r % capacity_;
        const size_t first = std::min(count, capacity_ - start);
        std::memcpy(out.data(), data_.data() + start, first * sizeof(float));
        if (first < count) {
            std::memcpy(out.data() + first, data_.data(), (count - first) * sizeof(float));
        }
        read_pos_.store(r + count, std::memory_order_release);
    }

    std::vector<float> data_;
    size_t capacity_ = 1;
    std::atomic<size_t> write_pos_{0};
    std::atomic<size_t> read_pos_{0};
    std::atomic<size_t> dropped_{0};
    std::mutex mutex_;  // Consumer side only; the producer never takes it.
    std::condition_variable cv_;
    bool closed_ = false;
};

}  // namespace pysysaudio

#endif  // PYSYSAUDIO_RING_BUFFER_H
