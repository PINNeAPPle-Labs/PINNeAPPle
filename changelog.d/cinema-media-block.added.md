- `experiment.json` gains an optional `cinema` block (schema stays `/1`, so every manifest already written remains
  valid): each video product made from an experiment records its file (or the repo and path it is hosted at), the
  SHA-256 of its bytes and its measured duration, width and height, so a clip traces back to a run instead of living
  as an unattributed file. `apps/<app>/cinema.json` carries the same block for apps, `pinneapple_studio.gallery`
  puts it in every gallery entry, and `manifest.check_cinema` (run by `scripts/check_experiment_manifests.py`) fails
  when a product names no bytes; set `PINNEAPPLE_CINEMA_ROOT` to a folder holding the media checkouts and it also
  compares each recorded hash against the file on disk.
