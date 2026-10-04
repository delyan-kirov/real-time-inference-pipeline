/**
 * @file gallery.hpp
 * @brief The enrolled gallery, and open-set matching against it.
 *
 * One template per identity: the normalised mean of that identity's still
 * embeddings. Mean-of-unit-vectors is the standard way to collapse several views
 * of a face into one template - more robust than picking a single "best" still,
 * and cheaper at match time than keeping every embedding and taking a maximum.
 *
 * Matching is deliberately *open-set*: the probe identity is not assumed to be
 * enrolled, so match() can return an unknown result. A closed-set matcher that
 * always names its nearest neighbour would report confident nonsense for every
 * face that walks past the camera.
 */

#pragma once

#include <opencv2/core.hpp>

#include "util.hpp"

namespace RFDxFACE {

/// RFD's vocabulary (Str, Vec, UInt, ...) is used unqualified throughout this
/// module; see util.hpp.
using namespace RFD;

/**
 * @brief The normalised mean of several embeddings - the gallery template for
 *        one identity.
 *
 * Exposed because enrollment needs to compare each still against this same mean
 * to spot mislabelled data, and a second implementation of it would be free to
 * disagree with Gallery::build().
 *
 * @param embeddings One or more 1x128 CV_32F unit vectors.
 * @return The renormalised mean, as a 1x128 CV_32F row.
 * @throws InvalidArgument on an empty input.
 */
[[nodiscard]] cv::Mat mean_template(const Vec<cv::Mat>& embeddings);

/**
 * @brief The outcome of matching one probe embedding against the gallery.
 */
struct Match {
    /// Empty when nothing cleared the threshold: the open-set rejection.
    Str identity;

    /**
     * @brief Cosine similarity to the best template, reported whether or not it
     *        won.
     *
     * Kept even on a rejection so callers can log near-misses.
     */
    float score = 0.0F;

    /// @return True when a template cleared the threshold.
    [[nodiscard]] bool known() const {
        return !identity.empty();
    }
};

/**
 * @brief One template per enrolled identity, built from stills and matched
 *        against live probes.
 */
class Gallery {
   public:
    // --- Building ------------------------------------------------------------

    /**
     * @brief Accumulates one embedding for an identity. Call build() when done.
     * @param identity  Non-empty identity name.
     * @param embedding A 1x128 CV_32F embedding.
     * @throws InvalidArgument on an empty identity or a malformed embedding.
     */
    void add(const Str& identity, const cv::Mat& embedding);

    /**
     * @brief Collapses accumulated embeddings into one normalised template per
     *        identity.
     *
     * Safe to call repeatedly; add() after build() needs another build() before
     * the new embedding is visible to match().
     */
    void build();

    // --- Matching ------------------------------------------------------------

    /**
     * @brief Finds the nearest template by cosine similarity.
     *
     * Templates and probes are both unit vectors, so the similarity is a dot
     * product.
     *
     * @param embedding A 1x128 CV_32F probe embedding.
     * @param threshold Minimum similarity to name an identity.
     * @return The best match, or an unknown Match when the best score is below
     *         @p threshold.
     * @throws InvalidArgument on a malformed embedding.
     */
    [[nodiscard]] Match match(const cv::Mat& embedding, float threshold) const;

    // --- Persistence ---------------------------------------------------------

    /**
     * @brief Writes templates plus the SFace digest they were produced with.
     * @param path Destination JSON file.
     * @throws RuntimeError if the gallery is empty or the file cannot be
     *         written.
     */
    void save(const Path& path) const;

    /**
     * @brief Loads a gallery written by save().
     * @param path Gallery JSON file.
     * @return The loaded gallery, ready to match against.
     * @throws RuntimeError if the file's model digest does not match the
     *         weights this binary was built against. That mismatch otherwise
     *         presents as mysteriously poor accuracy rather than as an error.
     */
    static Gallery load(const Path& path);

    // --- Inspection ----------------------------------------------------------

    /// @return How many identities are enrolled.
    [[nodiscard]] UInt size() const {
        return templates_.size();
    }

    /// @return Every enrolled identity name, in sorted order.
    [[nodiscard]] Vec<Str> identities() const;

    /**
     * @brief The template for an identity.
     *
     * Used by the evaluation tooling to print a full similarity matrix.
     *
     * @param identity An enrolled identity name.
     * @return That identity's 1x128 template.
     * @throws OutOfRange if the identity is absent.
     */
    [[nodiscard]] const cv::Mat& template_for(const Str& identity) const;

    /**
     * @brief How many stills went into an identity's template, for reporting.
     * @param identity An identity name.
     * @return The still count, or 0 if the identity is unknown.
     */
    [[nodiscard]] UInt source_count(const Str& identity) const;

   private:
    Map<Str, Vec<cv::Mat>> pending_;
    Map<Str, cv::Mat> templates_;
    Map<Str, UInt> source_counts_;
};

}  // namespace RFDxFACE
