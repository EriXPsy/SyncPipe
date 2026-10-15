# Generate the golden values that pin SyncPipe's native WCC surface to bsync.
#
# Run ONCE from the repository root by someone who has R and bsync installed:
#     Rscript tests/fixtures/bsync_golden/make_golden.R
# then commit the files it writes next to this script. tests/validation/test_bsync_conformance.py
# is skipped until golden_cells.csv exists and becomes a hard CI gate the moment it does.
#
# Parameters here MUST match tests/validation/test_bsync_conformance.py (window 30, lag_max 10, steps 1).
suppressPackageStartupMessages({
  library(bsync)
  library(generics)
})
dir <- "tests/fixtures/bsync_golden"
pair <- read.csv(file.path(dir, "pair.csv"))
args <- list(window_size = 30L, lag_max = 10L, window_increment = 1L, lag_increment = 1L)

for (stat in c("mean_abs_z", "peak")) {
  res <- do.call(wcc, c(list(pair$x, pair$y, statistic = stat), args))
  if (stat == "mean_abs_z") {
    write.csv(tidy(res), file.path(dir, "golden_cells.csv"), row.names = FALSE)  # columns: i, tau, wcc
  }
  write.csv(glance(res), file.path(dir, paste0("golden_glance_", stat, ".csv")), row.names = FALSE)
}
writeLines(
  c(paste("bsync", as.character(packageVersion("bsync"))), paste("R", as.character(getRversion()))),
  file.path(dir, "VERSIONS.txt")
)
message("Wrote golden values for bsync ", packageVersion("bsync"))
