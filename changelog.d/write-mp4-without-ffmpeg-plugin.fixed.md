`write_mp4` no longer fails when imageio is installed without its FFmpeg plugin (for example as a dependency of
scikit-image): imageio passed the frames to its TIFF plugin, which rejected the video arguments. It now uses imageio
only when `imageio-ffmpeg` is present and the `ffmpeg` program otherwise.
