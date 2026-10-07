/**
 * @file util.hpp
 * @brief RFD utililites
 *
 */

#pragma once

#include <array>
#include <atomic>
#include <chrono>
#include <cstddef>
#include <filesystem>
#include <fstream>
#include <map>
#include <memory>
#include <ostream>
#include <stdexcept>
#include <string>
#include <vector>

/**
 * @namespace RFD
 * @brief Project-wide vocabulary shared by every RFD module.
 */
namespace RFD {

// --- Text -------------------------------------------------------------------

/// An owning, mutable string. std::string.
using Str = std::string;

// --- Containers -------------------------------------------------------------

/// A contiguous, growable sequence. std::vector.
template <class T>
using Vec = std::vector<T>;

/// An ordered key-value container. std::map, so iteration is sorted by key -
/// which is what makes gallery output reproducible.
template <class K, class V>
using Map = std::map<K, V>;

/// A fixed-size sequence. std::array.
template <class T, std::size_t N>
using Arr = std::array<T, N>;

// --- Sizes ------------------------------------------------------------------

/// The unsigned type for sizes and indices. std::size_t.
using UInt = std::size_t;

// --- Ownership --------------------------------------------------------------

/// Sole ownership of a heap value. std::unique_ptr.
template <class T>
using Box = std::unique_ptr<T>;

// --- Concurrency ------------------------------------------------------------

/// A value safe to read and write from several threads. std::atomic.
template <class T>
using Atomic = std::atomic<T>;

// --- Filesystem and streams -------------------------------------------------

/// A filesystem path. std::filesystem::path.
using Path = std::filesystem::path;

/// An output stream to write text to, whatever is behind it.
using OStream = std::ostream;

/// A file opened for reading. std::ifstream.
using IFile = std::ifstream;

/// A file opened for writing. std::ofstream.
using OFile = std::ofstream;

// --- Time -------------------------------------------------------------------

/// The clock to measure elapsed time with. Monotonic, so it cannot run
/// backwards when the system clock is adjusted mid-stream.
using Clock = std::chrono::steady_clock;

/// A point in time from Clock.
using Instant = Clock::time_point;

/// A whole number of microseconds, the unit inference latency is reported in.
using Micros = std::chrono::microseconds;

/// A whole number of seconds.
using Seconds = std::chrono::seconds;

/// A fractional number of seconds, for wall-clock and throughput arithmetic.
using FloatSeconds = std::chrono::duration<double>;

// --- Errors -----------------------------------------------------------------

/// The base of everything we catch. std::exception.
using Exception = std::exception;

/// A failure that could not be foreseen from the arguments alone - a missing
/// file, a model that will not load. std::runtime_error.
using RuntimeError = std::runtime_error;

/// A caller passed something this function cannot accept. std::invalid_argument.
using InvalidArgument = std::invalid_argument;

/// A lookup key that is not present. std::out_of_range.
using OutOfRange = std::out_of_range;

}  // namespace RFD
