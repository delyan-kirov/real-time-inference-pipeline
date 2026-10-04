/**
 * @file rfd.cpp
 * @brief rfd - ingest a real-time video source, recognise faces in it, emit
 *        metadata.
 *
 * The inference is in include/face_filter.hpp

 * Pipeline properties:
 *
 *   identity sync=true      Force a file to stream like a live source
 *
 *   queue leaky=downstream  When inference cannot keep up, the queue discards the
 *                           oldest frame since we cannot block.
 *
 */

#include <gst/app/gstappsink.h>
#include <gst/gst.h>

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <iostream>
#include <nlohmann/json.hpp>
#include <string>

#include "face_meta.hpp"
#include "util.hpp"

/// RFD's vocabulary (Str, Vec, UInt, ...) is used unqualified in this file;
/// see util.hpp.
using namespace RFD;

/// Registers rfdface in-process; defined in plugin.cpp.
extern "C" gboolean rfd_register_static(void);

namespace {

/**
 * @brief Everything the command line can set.
 */
struct Options {
    /// File path, URI or V4L2 device to read from.
    Str source;
    /// Gallery JSON; empty means detection only.
    Str gallery;
    /// Where to write per-frame JSON; empty means stdout.
    Str output;
    /// Pace file sources to the clock.
    bool sync = true;
    /// Let the queue drop rather than block.
    bool drop = true;
    /// Leaky queue depth, in buffers.
    int queue_buffers = 2;
    /// Recognition threshold; <0 leaves the element's default.
    double threshold = -1.0;
    /// Stop after this many seconds; 0 runs until EOS.
    int max_seconds = 0;
    /// Suppress per-frame JSON and print only the summary.
    bool quiet = false;

    /**
     * @brief Whether --help was given.
     *
     * An explicit --help is a successful run, not a usage error. Tracked so main
     * can exit 0; returning failure breaks `rfd --help && ...` and makes any
     * wrapper think the program fell over.
     */
    bool help = false;
};

/**
 * @brief Prints the usage text to stderr.
 * @param argv0 The program name as invoked.
 */
void usage(const char* argv0) {
    std::cerr << "usage: " << argv0 << " --source <file|uri|/dev/videoN> [options]\n\n"
              << "  --source       file path, URI (rtsp://, file://, http://) or V4L2 device\n"
              << "  --gallery      gallery.json from rfd-enroll; omit for detection only\n"
              << "  --output       write JSON lines here instead of stdout\n"
              << "  --no-sync      do not pace to the clock; run as fast as possible\n"
              << "  --no-drop      make the queue block instead of dropping. Required for\n"
              << "                 a throughput measurement: with dropping on, the rate\n"
              << "                 measured is the drop rate, not inference capacity.\n"
              << "  --queue        leaky queue depth in buffers (default 2)\n"
              << "  --threshold    override the recognition threshold\n"
              << "  --seconds N    stop after N seconds\n"
              << "  --quiet        suppress per-frame JSON, print only the summary\n";
}

/**
 * @brief Parses argv into @p options.
 * @param argc    Argument count, as handed to main.
 * @param argv    Argument vector, as handed to main.
 * @param options Filled in on success; `help` is set when --help was given.
 * @return True when the options are usable; false on --help or a bad argument.
 */
bool parse_args(int argc, char* argv[], Options& options) {
    const Vec<Str> args(argv + 1, argv + argc);

    for (UInt i = 0; i < args.size(); ++i) {
        const Str& arg = args[i];
        auto value = [&](Str& out) {
            if (++i >= args.size()) {
                std::cerr << "error: " << arg << " needs a value\n";
                return false;
            }
            out = args[i];
            return true;
        };

        if (arg == "-h" || arg == "--help") {
            options.help = true;
            return false;
        } else if (arg == "--source") {
            if (!value(options.source)) return false;
        } else if (arg == "--gallery") {
            if (!value(options.gallery)) return false;
        } else if (arg == "--output") {
            if (!value(options.output)) return false;
        } else if (arg == "--no-sync") {
            options.sync = false;
        } else if (arg == "--no-drop") {
            options.drop = false;
        } else if (arg == "--quiet") {
            options.quiet = true;
        } else if (arg == "--queue") {
            Str raw;
            if (!value(raw)) return false;
            options.queue_buffers = std::stoi(raw);
        } else if (arg == "--threshold") {
            Str raw;
            if (!value(raw)) return false;
            options.threshold = std::stod(raw);
        } else if (arg == "--seconds") {
            Str raw;
            if (!value(raw)) return false;
            options.max_seconds = std::stoi(raw);
        } else {
            std::cerr << "error: unknown argument " << arg << '\n';
            return false;
        }
    }

    if (options.source.empty()) {
        std::cerr << "error: --source is required\n";
        return false;
    }
    return true;
}

/**
 * @brief Prints usage and picks the exit status.
 *
 * Only reached when parse_args failed, so the usage text goes to stderr; the
 * exit status distinguishes "you asked" from "you got it wrong".
 *
 * @param argv0   The program name as invoked.
 * @param options The partially parsed options, for its `help` flag.
 * @return EXIT_SUCCESS after --help, EXIT_FAILURE otherwise.
 */
int report_usage(const char* argv0, const Options& options) {
    usage(argv0);
    return options.help ? EXIT_SUCCESS : EXIT_FAILURE;
}

/**
 * @brief Whether the source produces frames at its own real-world rate.
 *
 * Cameras and network streams do; a file on disk is read as fast as the disk
 * allows.
 *
 * @param source The --source value.
 * @return True for a genuinely live source.
 */
bool is_live_source(const Str& source) {
    if (source.rfind("/dev/video", 0) == 0) {
        return true;
    }
    for (const char* scheme : {"rtsp://", "rtmp://", "udp://", "srt://", "rtp://"}) {
        if (source.rfind(scheme, 0) == 0) {
            return true;
        }
    }
    return false;
}

/**
 * @brief Picks the right GStreamer source for what the user actually passed.
 *
 * So one flag covers file, RTSP and webcam without the caller choosing an
 * element.
 *
 * @param source The --source value.
 * @return A gst-parse-launch fragment that produces raw video.
 */
Str source_description(const Str& source) {
    if (source.rfind("/dev/video", 0) == 0) {
        // A camera is genuinely live: do-timestamp makes the clock reflect capture
        // time rather than when buffers reached us.
        return "v4l2src device=" + source + " do-timestamp=true";
    }
    if (source.find("://") != Str::npos) {
        // uridecodebin covers rtsp://, file://, http:// and anything else
        // GStreamer has a source for.
        return "uridecodebin uri=" + source;
    }
    return "filesrc location=" + source + " ! decodebin";
}

/**
 * @brief Assembles the full pipeline description.
 * @param options The parsed command line.
 * @return A string for gst_parse_launch().
 */
Str build_pipeline(const Options& options) {
    Str description = source_description(options.source);

    description += " ! videoconvert ! video/x-raw,format=BGR";

    // Real-time pacing, ahead of the queue. Only for sources that are not already
    // live - see is_live_source and the file comment on why this cannot go at
    // the sink instead.
    if (options.sync && !is_live_source(options.source)) {
        description += " ! identity sync=true";
    }

    // The drop policy. leaky=downstream discards the oldest buffer when full;
    // leaky=no makes it block and propagate backpressure upstream.
    description +=
        options.drop ? " ! queue name=dropq leaky=downstream" : " ! queue name=dropq leaky=no";
    description += " max-size-buffers=" + std::to_string(options.queue_buffers);
    description += " max-size-bytes=0 max-size-time=0";

    description += " ! rfdface name=face";
    if (!options.gallery.empty()) {
        description += " gallery=" + options.gallery;
    }
    if (options.threshold >= 0.0) {
        description += " threshold=" + std::to_string(options.threshold);
    }

    // sync=false: the pipeline is already paced upstream, and syncing here too
    // would make the sink the bottleneck and corrupt the drop measurement.
    description += " ! appsink name=out max-buffers=4 drop=false sync=false";
    return description;
}

/// Frames entering the leaky queue. Atomic: the probe runs on the streaming
/// thread while main() reads it.
Atomic<guint64> frames_in{0};

/// Frames leaving the leaky queue.
Atomic<guint64> frames_out{0};

/// Buffer probe on the queue's sink pad.
GstPadProbeReturn count_in(GstPad*, GstPadProbeInfo*, gpointer) {
    frames_in.fetch_add(1, std::memory_order_relaxed);
    return GST_PAD_PROBE_OK;
}

/// Buffer probe on the queue's source pad.
GstPadProbeReturn count_out(GstPad*, GstPadProbeInfo*, gpointer) {
    frames_out.fetch_add(1, std::memory_order_relaxed);
    return GST_PAD_PROBE_OK;
}

/**
 * @brief Attaches buffer probes to both sides of the leaky queue.
 *
 * The difference between the two counters is exactly how many frames the queue
 * threw away, which is the number that says whether the pipeline kept up.
 *
 * @param pipeline The running pipeline, which must contain a queue named dropq.
 */
void attach_drop_probes(GstElement* pipeline) {
    GstElement* queue = gst_bin_get_by_name(GST_BIN(pipeline), "dropq");
    if (queue == nullptr) {
        return;
    }
    GstPad* sink_pad = gst_element_get_static_pad(queue, "sink");
    GstPad* src_pad = gst_element_get_static_pad(queue, "src");

    gst_pad_add_probe(sink_pad, GST_PAD_PROBE_TYPE_BUFFER, count_in, nullptr, nullptr);
    gst_pad_add_probe(src_pad, GST_PAD_PROBE_TYPE_BUFFER, count_out, nullptr, nullptr);

    gst_object_unref(sink_pad);
    gst_object_unref(src_pad);
    gst_object_unref(queue);
}

/**
 * @brief Serialises one analysed frame.
 *
 * Both the raw PTS and a zero-based timeline are reported, because they answer
 * different questions and can differ a lot. Our own clips start at PTS 4.2 s -
 * they were trimmed with timestamps preserved - so raw PTS is right for
 * correlating against the source file, and useless as "time since the stream
 * started". Emitting only PTS would push that confusion onto every consumer.
 *
 * @param buffer    The frame's buffer, for its PTS.
 * @param meta      The attached inference results.
 * @param first_pts The PTS of the first buffer seen, or GST_CLOCK_TIME_NONE.
 * @return One JSON object, ready to print as a line.
 */
nlohmann::json frame_to_json(GstBuffer* buffer, const GstRfdFaceMeta* meta,
                             GstClockTime first_pts) {
    nlohmann::json record;
    record["frame"] = meta->frame_number;

    if (GST_BUFFER_PTS_IS_VALID(buffer)) {
        const GstClockTime pts = GST_BUFFER_PTS(buffer);
        record["pts_ns"] = pts;
        record["t_ns"] =
            (first_pts != GST_CLOCK_TIME_NONE && pts >= first_pts) ? pts - first_pts : 0;
    } else {
        record["pts_ns"] = nullptr;
        record["t_ns"] = nullptr;
    }
    record["inference_us"] = meta->inference_us;
    record["faces"] = nlohmann::json::array();

    for (const RFDxGST::FaceResult& face : *meta->faces) {
        nlohmann::json entry;
        entry["box"] = {face.box.x, face.box.y, face.box.width, face.box.height};

        nlohmann::json landmarks = nlohmann::json::array();
        for (const cv::Point2f& point : face.landmarks) {
            landmarks.push_back({point.x, point.y});
        }
        entry["landmarks"] = std::move(landmarks);
        entry["detection_score"] = face.detection_score;

        // null rather than "" for an open-set rejection: a consumer checking for a
        // name should not have to know that empty string means "nobody".
        entry["identity"] =
            face.identified() ? nlohmann::json(face.identity) : nlohmann::json(nullptr);
        entry["similarity"] = face.similarity;
        record["faces"].push_back(std::move(entry));
    }
    return record;
}

/**
 * @brief Drains and reports every pending bus message.
 * @param bus The pipeline's bus.
 * @param eos Set to true if an end-of-stream message was seen.
 * @return False if the pipeline reported an error.
 */
bool drain_bus(GstBus* bus, bool& eos) {
    bool ok = true;
    while (GstMessage* message = gst_bus_pop_filtered(
               bus, static_cast<GstMessageType>(GST_MESSAGE_ERROR | GST_MESSAGE_EOS |
                                                GST_MESSAGE_WARNING))) {
        switch (GST_MESSAGE_TYPE(message)) {
            case GST_MESSAGE_ERROR: {
                GError* error = nullptr;
                gchar* debug = nullptr;
                gst_message_parse_error(message, &error, &debug);
                std::cerr << "pipeline error: " << error->message << '\n';
                if (debug != nullptr) {
                    std::cerr << "  debug: " << debug << '\n';
                }
                g_clear_error(&error);
                g_free(debug);
                ok = false;
                break;
            }
            case GST_MESSAGE_WARNING: {
                GError* error = nullptr;
                gst_message_parse_warning(message, &error, nullptr);
                std::cerr << "pipeline warning: " << error->message << '\n';
                g_clear_error(&error);
                break;
            }
            case GST_MESSAGE_EOS:
                eos = true;
                break;
            default:
                break;
        }
        gst_message_unref(message);
    }
    return ok;
}

}  // namespace

/**
 * @brief Builds the pipeline, pulls analysed frames and prints the summary.
 * @param argc Argument count.
 * @param argv Argument vector.
 * @return EXIT_SUCCESS on a clean run, EXIT_FAILURE on any pipeline error.
 */
int main(int argc, char* argv[]) {
    Options options;
    if (!parse_args(argc, argv, options)) {
        return report_usage(argv[0], options);
    }

    gst_init(&argc, &argv);

    // Registered in-process, so the binary does not depend on finding
    // libgstrfd.so at runtime. The plugin build exists for gst-launch-1.0.
    if (rfd_register_static() == FALSE) {
        std::cerr << "error: could not register the rfdface element\n";
        return EXIT_FAILURE;
    }

    const Str description = build_pipeline(options);
    if (!options.quiet) {
        std::cerr << "pipeline: " << description << "\n\n";
    }

    GError* error = nullptr;
    GstElement* pipeline = gst_parse_launch(description.c_str(), &error);
    if (pipeline == nullptr) {
        std::cerr << "error: " << (error != nullptr ? error->message : "cannot build pipeline")
                  << '\n';
        g_clear_error(&error);
        return EXIT_FAILURE;
    }

    attach_drop_probes(pipeline);

    GstElement* sink_element = gst_bin_get_by_name(GST_BIN(pipeline), "out");
    GstAppSink* sink = GST_APP_SINK(sink_element);

    OFile file;
    if (!options.output.empty()) {
        file.open(options.output);
        if (!file) {
            std::cerr << "error: cannot write " << options.output << '\n';
            gst_object_unref(sink_element);
            gst_object_unref(pipeline);
            return EXIT_FAILURE;
        }
    }
    OStream& out = options.output.empty() ? std::cout : file;

    if (gst_element_set_state(pipeline, GST_STATE_PLAYING) == GST_STATE_CHANGE_FAILURE) {
        std::cerr << "error: pipeline refused to start\n";
        gst_object_unref(sink_element);
        gst_object_unref(pipeline);
        return EXIT_FAILURE;
    }

    GstBus* bus = gst_element_get_bus(pipeline);
    const auto started = Clock::now();

    GstClockTime first_pts = GST_CLOCK_TIME_NONE;
    guint64 frames = 0;
    guint64 identified = 0;
    guint64 unknown = 0;
    Vec<guint64> inference_us;
    bool ok = true;
    bool eos = false;

    while (true) {
        if (!drain_bus(bus, eos)) {
            ok = false;
            break;
        }
        if (eos) {
            break;
        }
        if (options.max_seconds > 0) {
            const auto elapsed = Clock::now() - started;
            if (std::chrono::duration_cast<Seconds>(elapsed).count() >= options.max_seconds) {
                break;
            }
        }

        // Timed pull so the bus still gets serviced on a stalled or very slow
        // source rather than blocking here forever.
        GstSample* sample = gst_app_sink_try_pull_sample(sink, 100 * GST_MSECOND);
        if (sample == nullptr) {
            if (gst_app_sink_is_eos(sink)) {
                break;
            }
            continue;
        }

        GstBuffer* buffer = gst_sample_get_buffer(sample);
        if (buffer != nullptr) {
            const GstRfdFaceMeta* meta = gst_buffer_get_rfd_face_meta(buffer);
            if (meta != nullptr) {
                if (first_pts == GST_CLOCK_TIME_NONE && GST_BUFFER_PTS_IS_VALID(buffer)) {
                    first_pts = GST_BUFFER_PTS(buffer);
                }
                ++frames;
                inference_us.push_back(meta->inference_us);
                for (const RFDxGST::FaceResult& face : *meta->faces) {
                    if (face.identified()) {
                        ++identified;
                    } else {
                        ++unknown;
                    }
                }
                if (!options.quiet) {
                    out << frame_to_json(buffer, meta, first_pts).dump() << '\n';
                }
            }
        }
        gst_sample_unref(sample);
    }

    const auto elapsed = Clock::now() - started;
    const double seconds = FloatSeconds(elapsed).count();

    // Always return to NULL before dropping the last reference, or elements leak
    // the resources they allocated in PLAYING.
    gst_element_set_state(pipeline, GST_STATE_NULL);
    gst_object_unref(bus);
    gst_object_unref(sink_element);
    gst_object_unref(pipeline);

    const guint64 in = frames_in.load();
    const guint64 through = frames_out.load();
    const guint64 dropped = in > through ? in - through : 0;

    std::sort(inference_us.begin(), inference_us.end());
    const auto percentile = [&](double p) -> double {
        if (inference_us.empty()) return 0.0;
        const UInt index =
            std::min(inference_us.size() - 1,
                     static_cast<UInt>(p * static_cast<double>(inference_us.size())));
        return static_cast<double>(inference_us[index]) / 1000.0;
    };

    std::cerr << "\n--- summary ---\n"
              << "mode            " << (options.sync ? "real-time paced" : "unpaced") << ", queue "
              << (options.drop ? "leaky (drops)" : "blocking (no drops)") << '\n'
              << "wall clock      " << seconds << " s\n"
              << "frames decoded  " << in << '\n'
              << "frames analysed " << frames << '\n'
              << "frames dropped  " << dropped;
    if (in > 0) {
        std::cerr << "  (" << (100.0 * static_cast<double>(dropped) / static_cast<double>(in))
                  << "%)";
    }
    std::cerr << "\nidentified      " << identified << '\n'
              << "unknown         " << unknown << '\n';
    if (seconds > 0.0) {
        std::cerr << "throughput      " << static_cast<double>(frames) / seconds
                  << " fps analysed\n";
    }
    if (!inference_us.empty()) {
        std::cerr << "inference       median " << percentile(0.5) << " ms, p95 " << percentile(0.95)
                  << " ms, max " << percentile(1.0) << " ms\n";
    }

    return ok ? EXIT_SUCCESS : EXIT_FAILURE;
}
