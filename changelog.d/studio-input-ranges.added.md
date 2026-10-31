- Studio forms take the input box an author declares for a function (`fn.x_ranges = {"k": (lo, hi)}`, a literal at
  module level, read without importing): each pair becomes the input's `minimum` / `maximum`, and a name that is not a
  numeric input or a default outside its range is a form problem.
