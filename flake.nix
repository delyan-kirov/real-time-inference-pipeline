{
  description = "Real-time face-recognition inference pipeline built on GStreamer";

  inputs = {
    # Pinned by flake.lock. `nix flake metadata` prints the exact revision,
    # which is what fixes our GStreamer (and everything else) to an exact version.
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
  };

  outputs =
    { self, nixpkgs }:
    let
      systems = [
        "x86_64-linux"
        "aarch64-linux"
      ];
      forAllSystems = f: nixpkgs.lib.genAttrs systems (system: f (import nixpkgs { inherit system; }));
    in
    {
      devShells = forAllSystems (
        pkgs:
        let
          inherit (pkgs) lib stdenv;
          gst = pkgs.gst_all_1;

          # --- GStreamer ------------------------------------------------------
          # Core plus the plugin sets we actually need:
          #   base   - videoconvert, videoscale, videotestsrc, tee, queue, ...
          #   good   - v4l2src (webcam), rtpjitterbuffer, matroska, ...
          #   bad    - rtspsrc helpers, vaapi/va elements, codecs
          #   libav  - ffmpeg-backed decoders as a portable fallback
          #   rtsp-server - so we can synthesise RTSP feeds for load testing
          gstPackages = [
            gst.gstreamer
            gst.gst-plugins-base
            gst.gst-plugins-good
            gst.gst-plugins-bad
            gst.gst-plugins-ugly
            gst.gst-libav
            gst.gst-rtsp-server
            gst.gst-devtools
          ];

          # --- pkg-config wart -------------------------------------------------
          sysprofCaptureStub = pkgs.runCommand "sysprof-capture-4-stub" { } ''
            mkdir -p $out/lib/pkgconfig
            cat > $out/lib/pkgconfig/sysprof-capture-4.pc <<'PC'
            Name: sysprof-capture-4
            Description: Stub; see flake.nix. Private glib dependency, dynamic linking only.
            Version: 3.38.0
            Libs:
            Cflags:
            PC
            sed -i 's/^ *//' $out/lib/pkgconfig/sysprof-capture-4.pc
          '';

          # --- Intel hardware acceleration (VA-API) ---------------------------
          isIntelLinux = stdenv.hostPlatform.isLinux && stdenv.hostPlatform.isx86_64;
          vaapiPackages = lib.optionals isIntelLinux [
            pkgs.libva
            pkgs.libva-utils
            pkgs.intel-media-driver
          ];
        in
        {
          default = (pkgs.mkShell.override { stdenv = pkgs.gcc15Stdenv; }) {
            name = "rfd-dev";

            # Build tooling.
            nativeBuildInputs = with pkgs; [
              cmake
              gnumake
              pkg-config
              clang-tools # clangd, clang-format, clang-tidy
              gdb
              git
            ];

            buildInputs =
              gstPackages
              ++ vaapiPackages
              ++ [ sysprofCaptureStub ]
              ++ (with pkgs; [
                glib
                # Private (static-link-only) requirements of glib-2.0.pc and
                # gstreamer-1.0.pc. Listed so pkg-config resolves cleanly.
                pcre2
                libunwind
                elfutils
                zstd
                orc
                nlohmann_json
                spdlog
                opencv
                (python3.withPackages (
                  ps: with ps; [
                    numpy
                    opencv4
                  ]
                ))
              ]);

            shellHook = ''
              export GST_PLUGIN_SYSTEM_PATH_1_0="${
                lib.makeSearchPathOutput "lib" "lib/gstreamer-1.0" gstPackages
              }"

              # Keep the plugin registry cache inside the repo, not $HOME.
              export GST_REGISTRY_1_0="$PWD/.cache/gstreamer/registry.bin"
              mkdir -p "$PWD/.cache/gstreamer"

              ${lib.optionalString isIntelLinux ''
                # VA-API hardware decode on Intel graphics.
                export LIBVA_DRIVER_NAME=iHD
                export LIBVA_DRIVERS_PATH="${pkgs.intel-media-driver}/lib/dri"
              ''}

              # Let clangd/VS Code pick up the compilation database from ./build.
              export CMAKE_EXPORT_COMPILE_COMMANDS=1

              echo "rfd dev shell"
              echo "  gstreamer   $(pkg-config --modversion gstreamer-1.0)"
              echo "  cmake       $(cmake --version | head -n1 | cut -d' ' -f3)"
              echo "  compiler    $(c++ --version | head -n1)"
              echo "  make        $(make --version | head -n1)"
              echo
              echo "  configure:  cmake --preset dev"
              echo "  build:      cmake --build --preset dev"
              echo "  run:        ./build/dev/rfd-smoke"
            '';
          };
        }
      );
    };
}
