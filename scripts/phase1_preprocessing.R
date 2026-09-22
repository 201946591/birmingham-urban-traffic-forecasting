# ==============================================================================
# Phase 1: Data Engineering & Continuity Preprocessing in RStudio
# Urban Traffic Flow Forecasting in Birmingham (Local Authority 141)
# Author: Pretty Jayaraj (ID: 201936238) - MSc Business Analytics and Big Data
# University of Liverpool Management School - EBUS621 Dissertation Project
# ==============================================================================
# Transforms raw DfT manual survey records (72,948 rows) into the clean
# master hourly traffic panel: datetime parsing, continuity audit,
# cyclical sine/cosine transformations, correlation heatmap, and master CSV export.

library(tidyverse)
library(lubridate)
library(zoo)
library(corrplot)
library(readr)

cat("\n[INFO] Libraries loaded successfully.\n")

# ------------------------------------------------------------------------------
# STEP 1: LOAD & FILTER DATASET
# ------------------------------------------------------------------------------
raw_file_path <- "../dft_rawcount_local_authority_id_141.csv"

if (!file.exists(raw_file_path)) {
  # Fallback if script is run from project root directory
  raw_file_path <- "dft_rawcount_local_authority_id_141.csv"
}

cat(sprintf("\n[INFO] Loading dataset from: %s\n", raw_file_path))
df <- read_csv(raw_file_path, show_col_types = FALSE)

cat(sprintf("[INFO] Raw dataset dimension: %d rows x %d columns\n", nrow(df), ncol(df)))

# Isolate target variables and drop administrative/redundant columns
df_filtered <- df %>%
  select(
    count_point_id,
    direction_of_travel,
    year,
    count_date,
    hour,
    road_name,
    road_type,
    latitude,
    longitude,
    link_length_km,
    pedal_cycles,
    two_wheeled_motor_vehicles,
    cars_and_taxis,
    buses_and_coaches,
    lgvs,
    hgvs_2_rigid_axle,
    hgvs_3_rigid_axle,
    hgvs_4_or_more_rigid_axle,
    hgvs_3_or_4_articulated_axle,
    hgvs_5_articulated_axle,
    hgvs_6_articulated_axle,
    all_hgvs,
    all_motor_vehicles
  )

# ------------------------------------------------------------------------------
# STEP 2: DATETIME STRUCTURE (POSIXct Format)
# ------------------------------------------------------------------------------
cat("\n[INFO] Structuring datetime into POSIXct format...\n")

df_filtered <- df_filtered %>%
  mutate(
    # Combine date string and hour into explicit datetime POSIXct object
    datetime_str = sprintf("%s %02d:00:00", count_date, hour),
    datetime = as.POSIXct(datetime_str, format = "%Y-%m-%d %H:%M:%S", tz = "UTC")
  ) %>%
  select(-datetime_str)

# ------------------------------------------------------------------------------
# STEP 3: THE CONTINUITY AUDIT (diff() & Gaps Identification)
# ------------------------------------------------------------------------------
cat("\n[INFO] Performing Continuity Audit using chronological sorting & diff()...\n")

# Sort chronologically by count point, direction, and datetime
df_sorted <- df_filtered %>%
  arrange(count_point_id, direction_of_travel, datetime)

# Calculate hourly differences within each count_point + direction group
df_audit <- df_sorted %>%
  group_by(count_point_id, direction_of_travel) %>%
  mutate(
    time_diff_hours = as.numeric(difftime(datetime, lag(datetime), units = "hours"))
  ) %>%
  ungroup()

# Identify Gaps
gaps <- df_audit %>%
  filter(!is.na(time_diff_hours) & time_diff_hours > 1)

cat(sprintf("[AUDIT RESULT] Total timeline gaps detected: %d\n", nrow(gaps)))
cat(sprintf("[AUDIT RESULT] Gaps = 1 hour (normal): %d\n", sum(df_audit$time_diff_hours == 1, na.rm=TRUE)))
cat(sprintf("[AUDIT RESULT] Small gaps (2-3 hrs): %d\n", sum(df_audit$time_diff_hours %in% c(2,3), na.rm=TRUE)))
cat(sprintf("[AUDIT RESULT] Severe gaps (> 24 hrs): %d\n", sum(df_audit$time_diff_hours > 24, na.rm=TRUE)))

# Gap Treatment Strategy:
# For missing hours within count days, apply linear interpolation on numerical vehicle counts
# Severe multi-month gaps are preserved as distinct sampling episodes (inherent to DfT survey structure)
df_clean <- df_sorted %>%
  group_by(count_point_id, direction_of_travel, count_date) %>%
  mutate(
    across(
      c(pedal_cycles, two_wheeled_motor_vehicles, cars_and_taxis, buses_and_coaches, lgvs, all_hgvs, all_motor_vehicles),
      ~ if(sum(!is.na(.)) >= 2) zoo::na.approx(., na.rm = FALSE) else .
    )
  ) %>%
  ungroup()

# Replace any lingering NAs with column median for robustness
df_clean <- df_clean %>%
  mutate(across(where(is.numeric), ~ ifelse(is.na(.), median(., na.rm = TRUE), .)))

# ------------------------------------------------------------------------------
# STEP 4: CYCLICAL TRANSFORMATION (Sine & Cosine Encoding)
# ------------------------------------------------------------------------------
cat("\n[INFO] Engineering Cyclical Features (Sine & Cosine transformations)...\n")

df_encoded <- df_clean %>%
  mutate(
    month = month(datetime),
    day_of_week = wday(datetime, label = FALSE) - 1, # 0 = Sunday, 6 = Saturday
    is_weekend = ifelse(day_of_week %in% c(0, 6), 1, 0),
    
    # Hour cyclical encoding (period = 24)
    hour_sin = sin(2 * pi * hour / 24),
    hour_cos = cos(2 * pi * hour / 24),
    
    # Month cyclical encoding (period = 12)
    month_sin = sin(2 * pi * month / 12),
    month_cos = cos(2 * pi * month / 12),
    
    # Day of week cyclical encoding (period = 7)
    dow_sin = sin(2 * pi * day_of_week / 7),
    dow_cos = cos(2 * pi * day_of_week / 7)
  )

# ------------------------------------------------------------------------------
# STEP 5: DESCRIPTIVE STATISTICS & CORRELATION HEATMAP
# ------------------------------------------------------------------------------
cat("\n[INFO] Generating Correlation Heatmap for Dr. Ehsan...\n")

vehicle_vars <- c("pedal_cycles", "two_wheeled_motor_vehicles", 
                 "cars_and_taxis", "buses_and_coaches", "lgvs", 
                 "all_hgvs", "all_motor_vehicles")

cor_matrix <- cor(df_encoded %>% select(all_of(vehicle_vars)), use = "complete.obs")

# Print numerical correlation with all_motor_vehicles
cat("\n--- Correlation with Total Motor Vehicles (all_motor_vehicles) ---\n")
print(round(cor_matrix[, "all_motor_vehicles"], 4))

# Export Heatmap Plot to PNG
output_heatmap_path <- "correlation_heatmap.png"
png(filename = output_heatmap_path, width = 800, height = 700, res = 120)
corrplot(
  cor_matrix, 
  method = "color", 
  type = "upper", 
  addCoef.col = "black",
  number.cex = 0.8,
  tl.col = "black", 
  tl.srt = 45,
  title = "Birmingham Traffic: Vehicle Class Correlation Heatmap",
  mar = c(0, 0, 2, 0)
)
dev.off()
cat(sprintf("[INFO] Correlation heatmap saved to: %s\n", output_heatmap_path))

# ------------------------------------------------------------------------------
# STEP 6: EXPORT MASTER PREPROCESSED CSV
# ------------------------------------------------------------------------------
output_master_csv <- "master_birmingham_traffic.csv"
cat(sprintf("\n[INFO] Exporting preprocessed data to: %s...\n", output_master_csv))

write_csv(df_encoded, output_master_csv)

cat("\n==============================================================================\n")
cat("SUCCESS: Phase 1 Preprocessing Complete!\n")
cat(sprintf("Output File: %s (%d rows x %d columns)\n", output_master_csv, nrow(df_encoded), ncol(df_encoded)))
cat("==============================================================================\n")
