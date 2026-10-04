/**
 * @file gallery.cpp
 * @brief Implementation of RFDxFACE::Gallery - template building, open-set
 *        matching and the on-disk JSON format.
 */

#include "gallery.hpp"

#include <algorithm>
#include <nlohmann/json.hpp>
#include <string>

#include "embedder.hpp"

namespace RFDxFACE {

namespace {

/// Bumped whenever the on-disk layout changes; load() refuses anything else.
constexpr int FORMAT_VERSION = 1;

/// L2-normalises `row` into a fresh 1xN row that owns its storage.
cv::Mat normalised_copy(const cv::Mat& row) {
    cv::Mat out;
    cv::normalize(row, out, 1.0, 0.0, cv::NORM_L2);
    return out.reshape(1, 1).clone();
}

}  // namespace

void Gallery::add(const Str& identity, const cv::Mat& embedding) {
    if (identity.empty()) {
        throw InvalidArgument("Gallery::add needs a non-empty identity");
    }
    if (embedding.type() != CV_32F ||
        embedding.total() != static_cast<UInt>(Embedder::DIMENSIONS)) {
        throw InvalidArgument("Gallery::add expects a 1x128 CV_32F embedding");
    }
    pending_[identity].push_back(embedding.reshape(1, 1).clone());
}

cv::Mat mean_template(const Vec<cv::Mat>& embeddings) {
    if (embeddings.empty()) {
        throw InvalidArgument("mean_template needs at least one embedding");
    }

    cv::Mat sum = cv::Mat::zeros(1, Embedder::DIMENSIONS, CV_32F);
    for (const cv::Mat& embedding : embeddings) {
        sum += embedding.reshape(1, 1);
    }

    // Mean then renormalise. Dividing by the count is strictly redundant -
    // normalising the sum yields the same unit vector - but it keeps the intent
    // legible for anyone reading this as "the average face".
    sum /= static_cast<double>(embeddings.size());
    return normalised_copy(sum);
}

void Gallery::build() {
    for (const auto& [identity, embeddings] : pending_) {
        if (embeddings.empty()) {
            continue;
        }
        templates_[identity] = mean_template(embeddings);
        source_counts_[identity] = embeddings.size();
    }
    pending_.clear();
}

Match Gallery::match(const cv::Mat& embedding, float threshold) const {
    if (embedding.type() != CV_32F ||
        embedding.total() != static_cast<UInt>(Embedder::DIMENSIONS)) {
        throw InvalidArgument("Gallery::match expects a 1x128 CV_32F embedding");
    }

    const cv::Mat probe = embedding.reshape(1, 1);

    // Below the cosine range, so the first template always wins the comparison.
    Match best;
    best.score = -2.0F;

    for (const auto& [identity, templat] : templates_) {
        // Both sides are unit vectors, so the dot product *is* cosine similarity.
        const float score = static_cast<float>(probe.dot(templat));
        if (score > best.score) {
            best.score = score;
            best.identity = identity;
        }
    }

    // Report the best score either way, but clear the name on a rejection so
    // callers cannot accidentally treat a near-miss as an identification.
    if (best.score < threshold) {
        best.identity.clear();
    }
    return best;
}

void Gallery::save(const Path& path) const {
    if (templates_.empty()) {
        throw RuntimeError("refusing to save an empty gallery; call build() first");
    }

    nlohmann::json doc;
    doc["version"] = FORMAT_VERSION;
    doc["model"]["sface_sha256"] = Embedder::MODEL_DIGEST;
    doc["model"]["dimensions"] = Embedder::DIMENSIONS;
    doc["identities"] = nlohmann::json::array();

    for (const auto& [identity, templat] : templates_) {
        Vec<float> values(templat.begin<float>(), templat.end<float>());

        nlohmann::json entry;
        entry["name"] = identity;
        entry["sources"] = source_count(identity);
        entry["embedding"] = values;
        doc["identities"].push_back(std::move(entry));
    }

    OFile out(path);
    if (!out) {
        throw RuntimeError("cannot write gallery: " + path.string());
    }
    out << doc.dump(2) << '\n';
}

Gallery Gallery::load(const Path& path) {
    IFile in(path);
    if (!in) {
        throw RuntimeError("cannot read gallery: " + path.string());
    }

    const auto doc = nlohmann::json::parse(in);

    const auto version = doc.at("version").get<int>();
    if (version != FORMAT_VERSION) {
        throw RuntimeError("unsupported gallery format version " + std::to_string(version));
    }

    // An embedding only means anything relative to the weights that produced it.
    const auto digest = doc.at("model").at("sface_sha256").get<Str>();
    if (digest != Embedder::MODEL_DIGEST) {
        throw RuntimeError("gallery was built with different SFace weights\n  gallery: " + digest +
                           "\n  binary:  " + Str(Embedder::MODEL_DIGEST) +
                           "\nRe-run rfd-enroll against the current weights.");
    }

    Gallery gallery;
    for (const auto& entry : doc.at("identities")) {
        const auto name = entry.at("name").get<Str>();
        const auto values = entry.at("embedding").get<Vec<float>>();
        if (values.size() != static_cast<UInt>(Embedder::DIMENSIONS)) {
            throw RuntimeError("identity '" + name + "' has a malformed embedding");
        }

        // A Mat that owns its storage; `values` dies with this loop iteration.
        cv::Mat row(1, Embedder::DIMENSIONS, CV_32F);
        std::copy(values.begin(), values.end(), row.begin<float>());
        gallery.templates_[name] = row;
        gallery.source_counts_[name] = entry.value("sources", UInt{0});
    }

    if (gallery.templates_.empty()) {
        throw RuntimeError("gallery contains no identities: " + path.string());
    }
    return gallery;
}

Vec<Str> Gallery::identities() const {
    Vec<Str> names;
    names.reserve(templates_.size());
    for (const auto& entry : templates_) {
        names.push_back(entry.first);
    }
    return names;
}

const cv::Mat& Gallery::template_for(const Str& identity) const {
    const auto it = templates_.find(identity);
    if (it == templates_.end()) {
        throw OutOfRange("no such identity in gallery: " + identity);
    }
    return it->second;
}

UInt Gallery::source_count(const Str& identity) const {
    const auto it = source_counts_.find(identity);
    return it == source_counts_.end() ? 0 : it->second;
}

}  // namespace RFDxFACE
