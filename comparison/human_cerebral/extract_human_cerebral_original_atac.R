#!/usr/bin/env Rscript

# Extract the official original-cell cerebral-organoid ATAC matrix in a
# compact binary sparse format.  The downstream Python projection uses the
# exact peak order requested in frozen_peak_names.txt, so the large full
# matrix never needs to be serialized a second time.

suppressPackageStartupMessages({
  library(Matrix)
  library(GenomicRanges)
})

args <- commandArgs(trailingOnly = TRUE)
if (length(args) < 3L) {
  stop(
    "Usage: extract_human_cerebral_original_atac.R ",
    "<ATAC_all_merged_srt.rds> <frozen_peak_names.txt> <output_dir>"
  )
}

input_path <- normalizePath(args[[1]], mustWork = TRUE)
peak_path <- normalizePath(args[[2]], mustWork = TRUE)
output_dir <- normalizePath(args[[3]], mustWork = FALSE)
dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)

# The official ATAC experiment has no D12 or D18 libraries.  D16 is the only
# measured ATAC time between D11 and D21, so retain it for the raw-cell audit.
selected_days <- c(4L, 7L, 9L, 11L, 16L, 21L)

message("[read] ", input_path)
object <- readRDS(input_path)
# Access serialized slots through attributes. Calling Seurat accessors on this
# author-supplied object unnecessarily requires the full Seurat namespace.
object_attributes <- attributes(object)
object_class <- object_attributes[["class"]]
reduction_names <- names(object_attributes[["reductions"]])
metadata <- object_attributes[["meta.data"]]
metadata$cell_id <- rownames(metadata)

assays <- object_attributes[["assays"]]
assay_names <- names(assays)
assay_counts <- lapply(assays, function(assay) attributes(assay)[["counts"]])
assay_dims <- vapply(
  assay_counts,
  function(matrix) as.integer(attributes(matrix)[["Dim"]][[1]]),
  integer(1)
)
candidate_names <- assay_names[grepl("peak|atac", assay_names, ignore.case = TRUE)]
if (length(candidate_names) == 0L) {
  candidate_names <- assay_names[assay_dims == max(assay_dims)]
}
if (length(candidate_names) == 0L) {
  stop("Could not identify an ATAC/peaks assay")
}
assay_name <- candidate_names[[which.max(assay_dims[candidate_names])]]
counts <- assay_counts[[assay_name]]
if (is.null(counts)) {
  stop("Could not locate a counts layer in assay ", assay_name)
}
counts <- as(counts, "dgCMatrix")
cell_names <- attributes(counts)[["Dimnames"]][[2]]
if (!identical(rownames(metadata), cell_names)) {
  metadata_index <- match(cell_names, rownames(metadata))
  if (anyNA(metadata_index)) {
    stop("Counts columns cannot be aligned to metadata rows")
  }
  metadata <- metadata[metadata_index, , drop = FALSE]
}
rm(object, object_attributes, assays, assay_counts)
invisible(gc())

age_candidates <- c("age", "day", "time", "timepoint", "time_point")
age_column <- age_candidates[age_candidates %in% colnames(metadata)]
if (length(age_column) == 0L) {
  stop("No age/day metadata column found; available columns: ",
       paste(colnames(metadata), collapse = ", "))
}
age_column <- age_column[[1]]
age_text <- as.character(metadata[[age_column]])
age_match <- regexpr("[0-9]+", age_text)
age <- rep(NA_integer_, length(age_text))
valid_age <- age_match > 0L
age[valid_age] <- as.integer(regmatches(age_text, age_match)[valid_age])
if (anyNA(age)) {
  stop("Could not parse all values in age column ", age_column)
}

selected_cells <- which(age %in% selected_days)
if (length(selected_cells) == 0L) {
  stop("No cells matched selected days: ", paste(selected_days, collapse = ", "))
}

requested_peaks <- readLines(peak_path, warn = FALSE)
requested_peaks <- requested_peaks[nzchar(requested_peaks)]
if (anyDuplicated(requested_peaks)) {
  stop("Frozen peak list contains duplicated names")
}

parse_peaks <- function(values) {
  canonical <- gsub(":", "-", values, fixed = TRUE)
  parsed <- strcapture(
    "^(.*)-([0-9]+)-([0-9]+)$",
    canonical,
    proto = list(chromosome = character(), start = integer(), end = integer())
  )
  if (anyNA(parsed) || any(parsed$end <= parsed$start)) {
    stop("Could not parse genomic coordinates for all peak names")
  }
  parsed
}

source_peaks <- attributes(counts)[["Dimnames"]][[1]]
source_ranges <- parse_peaks(source_peaks)
requested_ranges <- parse_peaks(requested_peaks)
source_granges <- GRanges(
  source_ranges$chromosome,
  IRanges(start = source_ranges$start + 1L, end = source_ranges$end)
)
requested_granges <- GRanges(
  requested_ranges$chromosome,
  IRanges(start = requested_ranges$start + 1L, end = requested_ranges$end)
)
overlaps <- findOverlaps(requested_granges, source_granges, ignore.strand = TRUE)
overlap_map <- sparseMatrix(
  i = queryHits(overlaps),
  j = subjectHits(overlaps),
  x = 1,
  dims = c(length(requested_peaks), nrow(counts))
)
n_mapped_frozen <- length(unique(queryHits(overlaps)))
n_unmapped_frozen <- length(requested_peaks) - n_mapped_frozen

message(
  "[object] class=", paste(object_class, collapse = "/"),
  "; assays=", paste(assay_names, collapse = ","),
  "; selected_assay=", assay_name,
  "; full_shape=", nrow(counts), "x", ncol(counts)
)
message(
  "[subset] cells=", length(selected_cells),
  "; frozen_peaks=", length(requested_peaks),
  "; mapped_frozen_peaks=", n_mapped_frozen,
  "; overlap_edges=", length(overlaps),
  "; days=", paste(selected_days, collapse = ",")
)

# Re-bin the official peaks onto the frozen metacell peak intervals.  The two
# peak calls differ slightly in their boundaries, but 99%+ of frozen intervals
# have a genomic overlap. Process columns in chunks to avoid materializing a
# second multi-gigabyte sparse matrix in memory. The CSC slots are also the CSR
# slots of the transposed cell-by-feature matrix consumed by Python.
chunk_size <- as.integer(Sys.getenv("ATAC_EXTRACT_CHUNK_SIZE", "128"))
if (is.na(chunk_size) || chunk_size < 1L) {
  stop("ATAC_EXTRACT_CHUNK_SIZE must be a positive integer")
}
i_path <- file.path(output_dir, "original_atac_selected_peaks_csc_i.int32")
p_path <- file.path(output_dir, "original_atac_selected_peaks_csc_p.int32")
i_temp <- paste0(i_path, ".tmp")
p_temp <- paste0(p_path, ".tmp")
if (file.exists(i_path) || file.exists(p_path)) {
  stop("Refusing to overwrite an existing sparse export in ", output_dir)
}
unlink(c(i_temp, p_temp))
i_connection <- file(i_temp, open = "wb")
cumulative_p <- 0
all_p <- integer(length(selected_cells) + 1L)
output_column <- 0L
for (start in seq.int(1L, length(selected_cells), by = chunk_size)) {
  stop <- min(start + chunk_size - 1L, length(selected_cells))
  block_cells <- selected_cells[start:stop]
  mapped <- overlap_map %*% counts[, block_cells, drop = FALSE]
  mapped <- as(mapped, "dgCMatrix")
  mapped@x[] <- 1
  mapped <- drop0(mapped)
  writeBin(as.integer(mapped@i), i_connection, size = 4L, endian = "little")
  local_p <- as.numeric(mapped@p)
  n_block <- length(block_cells)
  all_p[(output_column + 2L):(output_column + n_block + 1L)] <-
    as.integer(cumulative_p + local_p[-1L])
  cumulative_p <- cumulative_p + length(mapped@i)
  output_column <- output_column + n_block
  message("[map] ", output_column, "/", length(selected_cells),
          " cells; nnz=", cumulative_p)
}
close(i_connection)
writeBin(all_p, p_temp, size = 4L, endian = "little")
if (!file.rename(i_temp, i_path) || !file.rename(p_temp, p_path)) {
  stop("Could not finalize sparse export files")
}

selected_metadata <- metadata[selected_cells, , drop = FALSE]
selected_metadata$cell_id <- rownames(selected_metadata)
selected_metadata$processed_age <- age[selected_cells]
if (!("n_cells_ATAC" %in% colnames(selected_metadata))) {
  selected_metadata$n_cells_ATAC <- 1L
}
if (!("nCount_peaks" %in% colnames(selected_metadata))) {
  selected_metadata$nCount_peaks <- Matrix::colSums(counts[, selected_cells, drop = FALSE])
}
if (!("nFeature_peaks" %in% colnames(selected_metadata))) {
  selected_metadata$nFeature_peaks <- diff(counts@p)[selected_cells]
}

metadata_connection <- gzfile(
  file.path(output_dir, "original_atac_cell_metadata.csv.gz"),
  open = "wt"
)
write.csv(selected_metadata, metadata_connection, row.names = FALSE)
close(metadata_connection)

writeLines(
  requested_peaks,
  file.path(output_dir, "original_atac_selected_peak_names.txt")
)
shape <- data.frame(
  field = c(
    "n_selected_peaks", "n_selected_cells", "nnz", "n_full_peaks",
    "n_full_cells", "assay", "age_column", "mapped_frozen_peaks",
    "unmapped_frozen_peaks", "overlap_edges"
  ),
  value = c(
    length(requested_peaks), length(selected_cells), cumulative_p, nrow(counts),
    ncol(counts), assay_name, age_column, n_mapped_frozen,
    n_unmapped_frozen, length(overlaps)
  )
)
write.table(
  shape,
  file.path(output_dir, "original_atac_sparse_shape.tsv"),
  sep = "\t",
  quote = FALSE,
  row.names = FALSE
)

day_counts <- as.data.frame(table(processed_age = age[selected_cells]))
colnames(day_counts)[[2]] <- "n_original_atac_cells"
write.csv(
  day_counts,
  file.path(output_dir, "original_atac_counts_by_day.csv"),
  row.names = FALSE
)

inventory <- c(
  paste0("object_class\t", paste(object_class, collapse = "/")),
  paste0("assays\t", paste(assay_names, collapse = ",")),
  paste0("selected_assay\t", assay_name),
  paste0("full_features\t", nrow(counts)),
  paste0("full_cells\t", ncol(counts)),
  paste0("selected_features\t", length(requested_peaks)),
  paste0("selected_cells\t", length(selected_cells)),
  paste0("selected_nnz\t", cumulative_p),
  paste0("mapped_frozen_peaks\t", n_mapped_frozen),
  paste0("unmapped_frozen_peaks\t", n_unmapped_frozen),
  paste0("overlap_edges\t", length(overlaps)),
  paste0("metadata_columns\t", paste(colnames(metadata), collapse = ",")),
  paste0("reductions\t", paste(reduction_names, collapse = ","))
)
writeLines(inventory, file.path(output_dir, "original_atac_object_inventory.tsv"))

message("[done] extracted sparse original-cell ATAC to ", output_dir)
