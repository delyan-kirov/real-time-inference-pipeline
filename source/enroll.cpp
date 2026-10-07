/**
 * @file enroll.cpp
 * @brief rfd-enroll - build a gallery from a directory of stills.
 *
 *     rfd-enroll <gallery-dir> [-o gallery.json] [--min-score N]
 *
 * Layout: one subdirectory per identity, images inside it.
 *
 *     gallery/
 *       reagan/portrait.jpg, press-1.jpg, ...
 *       nixon/...
 *
 */

#include <algorithm>
#include <cstdlib>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <opencv2/imgcodecs.hpp>

#include "detector.hpp"
#include "embedder.hpp"
#include "gallery.hpp"
#include "util.hpp"

/// RFD's vocabulary (Str, Vec, UInt, ...) is used unqualified in this file;
/// see util.hpp.
using namespace RFD;

namespace fs = std::filesystem;

namespace {

/**
 * @brief Everything the command line can set.
 */
struct Options {
    /// Directory holding one subdirectory per identity.
    fs::path gallery_dir;
    /// Where to write the gallery JSON.
    fs::path output = "gallery.json";
    /// Minimum detector confidence to accept a still.
    float min_score = 0.9F;

    /**
     * @brief Minimum cosine similarity between a still and its identity's mean.
     *
     * Set well below the matching threshold: the job here is to catch a different
     * person, not to reject an unusual pose of the right one.
     */
    float min_cohesion = 0.25F;

    /// An explicit --help is a successful run, not a usage error.
    bool help = false;
};

/**
 * @brief One embedded still, kept with its filename so warnings can name it.
 */
struct Still {
    /// The image's filename, without its directory.
    Str filename;
    /// Its 1x128 unit-length SFace embedding.
    cv::Mat embedding;
};

/**
 * @brief Prints the usage text to stderr.
 * @param argv0 The program name as invoked.
 */
void usage(const char* argv0) {
    std::cerr << "usage: " << argv0 << " <gallery-dir> [-o gallery.json] [--min-score 0.9]\n"
              << "\n"
              << "  <gallery-dir>  one subdirectory per identity, images inside\n"
              << "  -o             where to write the gallery (default gallery.json)\n"
              << "  --min-score    minimum detector confidence to accept (default 0.9)\n"
              << "  --min-cohesion minimum similarity to the identity mean (default 0.25)\n";
}

/**
 * @brief Whether a path looks like an image this tool can read.
 * @param path Any filesystem path.
 * @return True if the extension is one OpenCV decodes here.
 */
bool is_image(const fs::path& path) {
    static const Vec<Str> EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"};

    Str ext = path.extension().string();
    std::transform(ext.begin(), ext.end(), ext.begin(),
                   [](unsigned char c) { return static_cast<char>(std::tolower(c)); });

    return std::find(EXTENSIONS.begin(), EXTENSIONS.end(), ext) != EXTENSIONS.end();
}

/**
 * @brief Lists a directory's entries in sorted order.
 *
 * Sorted so enrollment is reproducible: directory iteration order is not.
 *
 * @param dir The directory to list.
 * @return Its entries, sorted by path.
 */
Vec<fs::path> sorted_children(const fs::path& dir) {
    Vec<fs::path> children;
    for (const auto& entry : fs::directory_iterator(dir)) {
        children.push_back(entry.path());
    }
    std::sort(children.begin(), children.end());
    return children;
}

/**
 * @brief Parses argv into @p options.
 * @param argc    Argument count, as handed to main.
 * @param argv    Argument vector, as handed to main.
 * @param options Filled in on success; `help` is set when --help was given.
 * @return True when the options are usable; false on --help or a bad argument.
 */
bool parse_args(int argc, char* argv[], Options& options) {
    Vec<Str> args(argv + 1, argv + argc);

    for (UInt i = 0; i < args.size(); ++i) {
        const Str& arg = args[i];

        if (arg == "-h" || arg == "--help") {
            options.help = true;
            return false;
        }
        if (arg == "-o" || arg == "--output") {
            if (++i >= args.size()) {
                std::cerr << "error: " << arg << " needs a value\n";
                return false;
            }
            options.output = args[i];
        } else if (arg == "--min-score") {
            if (++i >= args.size()) {
                std::cerr << "error: " << arg << " needs a value\n";
                return false;
            }
            options.min_score = std::stof(args[i]);
        } else if (arg == "--min-cohesion") {
            if (++i >= args.size()) {
                std::cerr << "error: " << arg << " needs a value\n";
                return false;
            }
            options.min_cohesion = std::stof(args[i]);
        } else if (!arg.empty() && arg.front() == '-') {
            std::cerr << "error: unknown option " << arg << '\n';
            return false;
        } else if (options.gallery_dir.empty()) {
            options.gallery_dir = arg;
        } else {
            std::cerr << "error: unexpected argument " << arg << '\n';
            return false;
        }
    }

    if (options.gallery_dir.empty()) {
        std::cerr << "error: no gallery directory given\n";
        return false;
    }
    return true;
}

/**
 * @brief Embeds every usable still in a directory.
 *
 * Images the detector cannot resolve unambiguously are reported and skipped.
 *
 * @param dir       One identity's directory of stills.
 * @param detector  The shared detector.
 * @param embedder  The shared embedder.
 * @param min_score Minimum detector confidence to accept a detection.
 * @return One Still per accepted image.
 */
Vec<Still> embed_stills(const fs::path& dir, RFDxFACE::Detector& detector,
                        const RFDxFACE::Embedder& embedder, float min_score) {
    Vec<Still> stills;

    for (const fs::path& image_path : sorted_children(dir)) {
        if (!fs::is_regular_file(image_path) || !is_image(image_path)) {
            continue;
        }

        const cv::Mat image = cv::imread(image_path.string(), cv::IMREAD_COLOR);
        if (image.empty()) {
            std::cerr << "  skip " << image_path.filename().string() << ": cannot decode\n";
            continue;
        }

        auto detections = detector.detect(image);

        // Drop low-confidence detections before counting, so a spurious
        // background face does not make a perfectly good portrait "ambiguous".
        std::erase_if(detections,
                      [min_score](const RFDxFACE::Detection& d) { return d.score < min_score; });

        if (detections.empty()) {
            std::cerr << "  skip " << image_path.filename().string() << ": no face found\n";
            continue;
        }
        if (detections.size() > 1) {
            std::cerr << "  skip " << image_path.filename().string() << ": " << detections.size()
                      << " faces found, cannot tell which is the subject\n";
            continue;
        }

        stills.push_back(
            {image_path.filename().string(), embedder.embed(image, detections.front().row)});

        std::cout << "  ok   " << image_path.filename().string() << "  (score "
                  << detections.front().score << ")\n";
    }

    return stills;
}

/**
 * @brief Drops stills that disagree with the identity's own mean.
 *
 * See the file comment for what this catches. The mean is recomputed after
 * dropping, because the outlier contributed to the mean it was measured against.
 * One pass is enough for the single-mislabel case this guards against; a folder
 * that is mostly the wrong person is a data problem no heuristic should paper
 * over, and it will show up in the separation matrix.
 *
 * @param stills       One identity's embedded stills.
 * @param min_cohesion Minimum similarity to the identity mean.
 * @return The survivors.
 */
Vec<Still> drop_outliers(Vec<Still> stills, float min_cohesion) {
    if (stills.size() < 3) {
        // With one or two stills there is no majority to disagree with.
        return stills;
    }

    Vec<cv::Mat> embeddings;
    embeddings.reserve(stills.size());
    for (const Still& still : stills) {
        embeddings.push_back(still.embedding);
    }
    const cv::Mat mean = RFDxFACE::mean_template(embeddings);

    Vec<Still> kept;
    for (const Still& still : stills) {
        const float similarity = static_cast<float>(still.embedding.dot(mean));
        if (similarity < min_cohesion) {
            std::cerr << "  DROP " << still.filename << ": similarity " << similarity
                      << " to this identity's mean is below " << min_cohesion
                      << " - probably a different person\n";
            continue;
        }
        kept.push_back(still);
    }
    return kept;
}

/**
 * @brief Prints between-identity similarity and each identity's worst internal
 *        agreement.
 *
 * Verifies the thing that actually matters for open-set matching: the weakest
 * within-identity score must beat the strongest between-identity score, or no
 * single threshold can separate them.
 *
 * @param gallery     The built gallery.
 * @param by_identity The stills each identity's template was built from.
 */
void report_separation(const RFDxFACE::Gallery& gallery, const Map<Str, Vec<Still>>& by_identity) {
    const Vec<Str> names = gallery.identities();

    std::cout << "\nbetween-identity cosine similarity\n      ";
    for (const Str& name : names) {
        std::cout << std::setw(9) << name.substr(0, 8);
    }
    std::cout << '\n';

    float worst_between = -1.0F;
    for (const Str& row : names) {
        std::cout << std::setw(8) << row.substr(0, 8);
        for (const Str& col : names) {
            const float similarity =
                static_cast<float>(gallery.template_for(row).dot(gallery.template_for(col)));
            if (row != col) {
                worst_between = std::max(worst_between, similarity);
            }
            std::cout << std::setw(9) << std::fixed << std::setprecision(3) << similarity;
        }
        std::cout << '\n';
    }

    std::cout << "\nwithin-identity agreement (each still vs its own template)\n";
    float worst_within = 1.0F;
    for (const Str& name : names) {
        float lowest = 1.0F;
        for (const Still& still : by_identity.at(name)) {
            lowest = std::min(lowest,
                              static_cast<float>(still.embedding.dot(gallery.template_for(name))));
        }
        worst_within = std::min(worst_within, lowest);
        std::cout << "  " << std::setw(8) << name << "  worst " << std::fixed
                  << std::setprecision(3) << lowest << "  over " << by_identity.at(name).size()
                  << " stills\n";
    }

    const float margin = worst_within - worst_between;
    std::cout << "\nseparation margin " << std::fixed << std::setprecision(3) << margin
              << "  (worst within " << worst_within << " - best between " << worst_between << ")\n";
    if (margin <= 0.0F) {
        std::cout << "  WARNING: the sets overlap; no single threshold separates them\n";
    }
}

}  // namespace

/**
 * @brief Enrolls every identity under the given directory and writes the gallery.
 * @param argc Argument count.
 * @param argv Argument vector.
 * @return EXIT_SUCCESS when a gallery was written, EXIT_FAILURE otherwise.
 */
int main(int argc, char* argv[]) {
    Options options;
    if (!parse_args(argc, argv, options)) {
        usage(argv[0]);
        return options.help ? EXIT_SUCCESS : EXIT_FAILURE;
    }

    try {
        if (!fs::is_directory(options.gallery_dir)) {
            std::cerr << "error: not a directory: " << options.gallery_dir.string() << '\n';
            return EXIT_FAILURE;
        }

        RFDxFACE::Detector detector(options.min_score);
        const RFDxFACE::Embedder embedder;
        RFDxFACE::Gallery gallery;

        Map<Str, Vec<Still>> by_identity;
        UInt total_stills = 0;

        for (const fs::path& identity_dir : sorted_children(options.gallery_dir)) {
            if (!fs::is_directory(identity_dir)) {
                continue;
            }

            const Str identity = identity_dir.filename().string();
            std::cout << identity << '\n';

            Vec<Still> stills =
                drop_outliers(embed_stills(identity_dir, detector, embedder, options.min_score),
                              options.min_cohesion);

            if (stills.empty()) {
                std::cerr << "  WARNING: no usable stills, identity not enrolled\n";
                continue;
            }

            for (const Still& still : stills) {
                gallery.add(identity, still.embedding);
            }
            total_stills += stills.size();
            by_identity.emplace(identity, std::move(stills));
        }

        if (by_identity.empty()) {
            std::cerr << "error: nothing enrolled; expected <gallery-dir>/<identity>/*.jpg\n";
            return EXIT_FAILURE;
        }

        gallery.build();
        report_separation(gallery, by_identity);
        gallery.save(options.output);

        std::cout << "\nenrolled " << by_identity.size() << " identities from " << total_stills
                  << " stills -> " << options.output.string() << '\n'
                  << "sface weights " << RFDxFACE::Embedder::MODEL_DIGEST << '\n';

    } catch (const Exception& e) {
        std::cerr << "error: " << e.what() << '\n';
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}
