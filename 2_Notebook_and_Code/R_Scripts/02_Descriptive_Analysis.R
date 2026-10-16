# ============================================================
# IT3091 MACHINE LEARNING - FOOTBALL ANALYTICS PROJECT
# FILE: 02_Descriptive_Analysis.R   (corrected version)
#
# PURPOSE:
# Descriptive analysis / EDA on the final engineered dataset
# BEFORE ML modelling.
#
# DATASET:
# European_Football_PlayerAware_Primary_9Leagues_2008_2026.csv
#
# WHAT CHANGED COMPARED WITH THE FIRST VERSION
# --------------------------------------------
# Any step that relates a feature to the OUTCOME or that helps
# decide which features to use (Steps 21, 22, 27, 28, 29) now uses
# TRAINING-PERIOD matches only (season_start <= 2022), stored in
# `train_data`. Validation (2023) and test (2024+) matches are
# therefore not inspected for feature-outcome relationships.
#
# Steps that only describe the whole file (counts, QA, missingness,
# source comparison) still use all matches, because they do not
# influence modelling decisions. Steps 8 and 12 show the outcome
# mix over ALL seasons for reporting; do not use them to choose
# features or models.
#
# BUG FIX: Step 28 failed with "invalid 'na.print' specification"
# because print(..., n = Inf) was called on a plain data.frame.
# The correlation table is now a tibble.
# ============================================================


# ============================================================
# STEP 0 - INSTALL PACKAGES (run once if needed)
# ============================================================
# install.packages("tidyverse")
# install.packages("corrplot")


# ============================================================
# STEP 1 - LOAD LIBRARIES
# ============================================================
library(tidyverse)
library(corrplot)


# ============================================================
# STEP 2 - LOAD FINAL ENGINEERED DATASET
# ============================================================
# Change this path if your CSV is in another folder.
file_path <- "European_Football_PlayerAware_Primary_9Leagues_2008_2026.csv"

football <- read_csv(file_path, show_col_types = FALSE)


# ============================================================
# STEP 3 - REMOVE ACCIDENTAL CSV INDEX COLUMN
# ============================================================
# Sometimes pandas writes an extra "index" column. It is not an
# analytical feature.
football <- football %>% select(-any_of("index"))


# ============================================================
# STEP 4 - FIX DATA TYPES AND CREATE THE TRAINING-ONLY DATASET
# ============================================================
football <- football %>%
  mutate(
    date          = as.Date(date),
    target_result = factor(target_result, levels = c("H", "D", "A")),
    league_code   = factor(league_code),
    data_source   = factor(data_source),
    season_start  = as.integer(season_start)
  )

# Training period used for every outcome-related / feature-selection
# step below (matches the ML notebook: season_start <= 2022).
train_data <- football %>% filter(season_start <= 2022)

cat("All matches     :", nrow(football), "\n")
cat("Training matches:", nrow(train_data), "\n")


# ============================================================
# STEP 5 - BASIC DATASET UNDERSTANDING
# ============================================================
cat("\n==============================\n")
cat("DATASET BASIC INFORMATION\n")
cat("==============================\n")

cat("Number of rows:", nrow(football), "\n")
n_columns_original <- ncol(football)   # before helper columns such as target_check are added
cat("Number of columns:", n_columns_original, "\n")
cat("Date range:",
    as.character(min(football$date, na.rm = TRUE)), "to",
    as.character(max(football$date, na.rm = TRUE)), "\n")
cat("Number of leagues:", n_distinct(football$league_code), "\n")
cat("Number of seasons:", n_distinct(football$season_start), "\n")

glimpse(football)
names(football)
head(football)


# ============================================================
# STEP 6 - CHECK DUPLICATES
# ============================================================
duplicate_match_uid <- sum(duplicated(football$match_uid))
duplicate_game_id   <- sum(duplicated(football$game_id))

cat("\nDuplicate match_uid:", duplicate_match_uid, "\n")
cat("Duplicate game_id:", duplicate_game_id, "\n")


# ============================================================
# STEP 7 - CHECK TARGET CONSISTENCY
# ============================================================
# Reconstruct H/D/A from the final goals.
# IMPORTANT: goals are NOT ML predictors. They are used here only
# to verify the target.
football <- football %>%
  mutate(
    target_check = case_when(
      audit_home_goals >  audit_away_goals ~ "H",
      audit_home_goals <  audit_away_goals ~ "A",
      audit_home_goals == audit_away_goals ~ "D",
      TRUE ~ NA_character_
    )
  )

target_mismatch <- sum(football$target_check != as.character(football$target_result),
                       na.rm = TRUE)
cat("\nTarget mismatches:", target_mismatch, "\n")


# ============================================================
# STEP 8 - TARGET DISTRIBUTION (all matches, reporting only)
# ============================================================
target_distribution <- football %>%
  count(target_result) %>%
  mutate(percentage = 100 * n / sum(n))

print(target_distribution)

ggplot(target_distribution, aes(x = target_result, y = n)) +
  geom_col() +
  labs(title = "Distribution of Match Outcomes",
       x = "Match Outcome", y = "Number of Matches") +
  theme_minimal()


# ============================================================
# STEP 9 - LEAGUE DISTRIBUTION
# ============================================================
league_distribution <- football %>%
  count(league_code, sort = TRUE) %>%
  mutate(percentage = 100 * n / sum(n))

print(league_distribution)

ggplot(league_distribution, aes(x = reorder(league_code, n), y = n)) +
  geom_col() +
  coord_flip() +
  labs(title = "Number of Matches by League",
       x = "League", y = "Number of Matches") +
  theme_minimal()


# ============================================================
# STEP 10 - SOURCE DISTRIBUTION
# ============================================================
source_distribution <- football %>%
  count(data_source) %>%
  mutate(percentage = 100 * n / sum(n))

print(source_distribution)


# ============================================================
# STEP 11 - SEASON-BY-SEASON MATCH COUNTS
# ============================================================
season_counts <- football %>% count(season_start)

print(season_counts)

ggplot(season_counts, aes(x = season_start, y = n)) +
  geom_line() +
  geom_point() +
  labs(title = "Number of Matches by Season",
       x = "Season Start Year", y = "Number of Matches") +
  theme_minimal()


# ============================================================
# STEP 12 - SEASON-BY-SEASON OUTCOME DISTRIBUTION
#           (all seasons, reporting only)
# ============================================================
season_outcomes <- football %>%
  count(season_start, target_result) %>%
  group_by(season_start) %>%
  mutate(percentage = 100 * n / sum(n)) %>%
  ungroup()

print(season_outcomes)

ggplot(season_outcomes, aes(x = season_start, y = percentage, colour = target_result)) +
  geom_line(linewidth = 1) +
  geom_point() +
  labs(title = "Match Outcome Percentage by Season",
       x = "Season Start Year", y = "Percentage", colour = "Outcome") +
  theme_minimal()


# ============================================================
# STEP 13 - MISSING VALUE ANALYSIS
# ============================================================
missing_summary <- tibble(
  variable           = names(football),
  missing_count      = sapply(football, function(x) sum(is.na(x))),
  missing_percentage = sapply(football, function(x) mean(is.na(x)) * 100)
) %>%
  arrange(desc(missing_percentage))

print(missing_summary, n = Inf)

# Plot only variables that have missing values
missing_summary %>%
  filter(missing_percentage > 0) %>%
  ggplot(aes(x = reorder(variable, missing_percentage), y = missing_percentage)) +
  geom_col() +
  coord_flip() +
  labs(title = "Missing Values by Variable",
       x = "Feature", y = "Missing Percentage") +
  theme_minimal()


# ============================================================
# STEP 14 - PLAYER FEATURE COVERAGE
# ============================================================
player_ready_summary <- football %>%
  count(player_feature_ready) %>%
  mutate(percentage = 100 * n / sum(n))

print(player_ready_summary)

coverage_by_season <- football %>%
  group_by(season_start) %>%
  summarise(
    matches = n(),
    player_ready_percentage = mean(player_feature_ready, na.rm = TRUE) * 100,
    .groups = "drop"
  )

print(coverage_by_season)

ggplot(coverage_by_season, aes(x = season_start, y = player_ready_percentage)) +
  geom_line() +
  geom_point() +
  labs(title = "Player Feature Availability by Season",
       x = "Season", y = "Player-ready Matches (%)") +
  theme_minimal()


# ============================================================
# STEP 15 - SELECT IMPORTANT NUMERIC FEATURES
# ============================================================
important_numeric <- c(
  "home_points_per_game_L5", "away_points_per_game_L5",
  "home_win_rate_L5", "away_win_rate_L5",
  "home_goals_for_avg_L5", "away_goals_for_avg_L5",
  "home_goals_against_avg_L5", "away_goals_against_avg_L5",
  "home_goal_diff_avg_L5", "away_goal_diff_avg_L5",
  "home_xi_strength", "away_xi_strength",
  "home_attack_strength", "away_attack_strength",
  "home_midfield_strength", "away_midfield_strength",
  "home_defence_strength", "away_defence_strength",
  "home_gk_strength", "away_gk_strength",
  "home_top3_strength", "away_top3_strength",
  "home_top3_gap", "away_top3_gap",
  "home_lineup_continuity", "away_lineup_continuity",
  "xi_strength_diff",
  "home_attack_vs_away_defence", "away_attack_vs_home_defence",
  "midfield_strength_diff", "gk_strength_diff"
)

# Keep only columns actually present
important_numeric <- intersect(important_numeric, names(football))


# ============================================================
# STEP 16 - FUNCTION TO CALCULATE SKEWNESS
# ============================================================
# Skewness shows whether a distribution is symmetric or strongly
# left/right skewed (sample skewness, adjusted Fisher-Pearson).
skewness_manual <- function(x) {
  x <- x[is.finite(x)]
  n <- length(x)
  if (n < 3 || sd(x) == 0) {
    return(NA_real_)
  }
  x_mean <- mean(x)
  x_sd   <- sd(x)
  (n / ((n - 1) * (n - 2))) * sum(((x - x_mean) / x_sd)^3)
}


# ============================================================
# STEP 17 - FULL DESCRIPTIVE STATISTICS
# ============================================================
descriptive_statistics <- football %>%
  select(all_of(important_numeric)) %>%
  pivot_longer(cols = everything(), names_to = "feature", values_to = "value") %>%
  group_by(feature) %>%
  summarise(
    valid_n            = sum(!is.na(value)),
    missing_n          = sum(is.na(value)),
    mean               = mean(value, na.rm = TRUE),
    median             = median(value, na.rm = TRUE),
    variance           = var(value, na.rm = TRUE),
    standard_deviation = sd(value, na.rm = TRUE),
    minimum            = min(value, na.rm = TRUE),
    Q1                 = quantile(value, 0.25, na.rm = TRUE),
    Q3                 = quantile(value, 0.75, na.rm = TRUE),
    IQR                = IQR(value, na.rm = TRUE),
    maximum            = max(value, na.rm = TRUE),
    skewness           = skewness_manual(value),
    .groups = "drop"
  )

print(descriptive_statistics, n = Inf)


# ============================================================
# STEP 18 - SAVE DESCRIPTIVE STATISTICS
# ============================================================
write_csv(descriptive_statistics, "descriptive_statistics.csv")
write_csv(missing_summary,        "missing_value_summary.csv")
write_csv(target_distribution,    "target_distribution.csv")
write_csv(league_distribution,    "league_distribution.csv")


# ============================================================
# STEP 19 - XI STRENGTH DISTRIBUTIONS
# ============================================================
ggplot(football, aes(x = home_xi_strength)) +
  geom_histogram(bins = 40) +
  labs(title = "Distribution of Home XI Strength",
       x = "Home XI Strength", y = "Frequency") +
  theme_minimal()

ggplot(football, aes(x = away_xi_strength)) +
  geom_histogram(bins = 40) +
  labs(title = "Distribution of Away XI Strength",
       x = "Away XI Strength", y = "Frequency") +
  theme_minimal()


# ============================================================
# STEP 20 - XI STRENGTH DIFFERENCE DISTRIBUTION
# ============================================================
ggplot(football, aes(x = xi_strength_diff)) +
  geom_histogram(bins = 50) +
  labs(title = "Distribution of XI Strength Difference",
       x = "Home XI Strength - Away XI Strength", y = "Frequency") +
  theme_minimal()


# ============================================================
# STEP 21 - XI STRENGTH DIFFERENCE BY OUTCOME  (TRAINING ONLY)
# ============================================================
ggplot(train_data, aes(x = target_result, y = xi_strength_diff)) +
  geom_boxplot() +
  labs(title = "XI Strength Difference by Match Outcome (training period)",
       x = "Match Outcome", y = "XI Strength Difference") +
  theme_minimal()


# ============================================================
# STEP 22 - DENSITY OF XI DIFFERENCE BY OUTCOME  (TRAINING ONLY)
# ============================================================
ggplot(train_data, aes(x = xi_strength_diff, colour = target_result)) +
  geom_density(na.rm = TRUE) +
  labs(title = "XI Strength Difference Distribution by Outcome (training period)",
       x = "XI Strength Difference", y = "Density", colour = "Outcome") +
  theme_minimal()


# ============================================================
# STEP 23 - RECENT FORM DISTRIBUTION
# ============================================================
ggplot(football, aes(x = home_points_per_game_L5)) +
  geom_histogram(bins = 30) +
  labs(title = "Distribution of Home Recent Points per Game",
       x = "Home PPG - Last 5", y = "Frequency") +
  theme_minimal()


# ============================================================
# STEP 24 - LINEUP CONTINUITY DISTRIBUTION
# ============================================================
ggplot(football, aes(x = home_lineup_continuity)) +
  geom_histogram(bins = 25) +
  labs(title = "Distribution of Home Lineup Continuity",
       x = "Lineup Continuity", y = "Frequency") +
  theme_minimal()


# ============================================================
# STEP 25 - MOST COMMON FORMATIONS
# ============================================================
home_formations <- football %>%
  filter(!is.na(home_formation)) %>%
  count(home_formation, sort = TRUE) %>%
  slice_head(n = 15)

print(home_formations)

ggplot(home_formations, aes(x = reorder(home_formation, n), y = n)) +
  geom_col() +
  coord_flip() +
  labs(title = "15 Most Common Home Formations",
       x = "Formation", y = "Number of Matches") +
  theme_minimal()


# ============================================================
# STEP 26 - STAGE 1 VS STAGE 2 COMPARISON
# ============================================================
source_comparison <- football %>%
  group_by(data_source) %>%
  summarise(
    matches = n(),
    mean_home_xi = mean(home_xi_strength, na.rm = TRUE),
    sd_home_xi   = sd(home_xi_strength, na.rm = TRUE),
    mean_away_xi = mean(away_xi_strength, na.rm = TRUE),
    player_ready_percentage = 100 * mean(player_feature_ready, na.rm = TRUE),
    .groups = "drop"
  )

print(source_comparison)

# Density comparison - descriptive only
ggplot(football, aes(x = home_xi_strength, colour = data_source)) +
  geom_density(na.rm = TRUE) +
  labs(title = "Home XI Strength Distribution by Data Source",
       x = "Home XI Strength", y = "Density", colour = "Source") +
  theme_minimal()


# ============================================================
# STEP 27 - CORRELATION MATRIX  (TRAINING ONLY)
# ============================================================
# Correlations between features guide multicollinearity decisions,
# so they are computed on the training period only.
correlation_features <- c(
  "home_points_per_game_L5", "away_points_per_game_L5",
  "home_goal_diff_avg_L5", "away_goal_diff_avg_L5",
  "home_xi_strength", "away_xi_strength",
  "home_top3_strength", "away_top3_strength",
  "home_attack_strength", "away_attack_strength",
  "home_midfield_strength", "away_midfield_strength",
  "home_defence_strength", "away_defence_strength",
  "home_gk_strength", "away_gk_strength",
  "xi_strength_diff",
  "home_attack_vs_away_defence", "away_attack_vs_home_defence",
  "midfield_strength_diff", "gk_strength_diff"
)

correlation_features <- intersect(correlation_features, names(train_data))

correlation_data <- train_data %>% select(all_of(correlation_features))

pearson_correlation <- cor(correlation_data,
                           use = "pairwise.complete.obs",
                           method = "pearson")

print(round(pearson_correlation, 3))

# Heatmap
corrplot(pearson_correlation, method = "color", type = "upper", tl.cex = 0.7)

# Save correlation matrix
write.csv(pearson_correlation, "pearson_correlation_matrix.csv")


# ============================================================
# STEP 28 - FIND HIGHLY CORRELATED FEATURE PAIRS (|r| >= 0.80)
# ============================================================
# as_tibble() is needed so that print(..., n = Inf) works (a plain data.frame
# would raise "invalid 'na.print' specification").
cor_table <- as_tibble(as.data.frame(as.table(pearson_correlation)))
names(cor_table) <- c("feature_1", "feature_2", "correlation")

high_correlations <- cor_table %>%
  filter(feature_1 != feature_2, abs(correlation) >= 0.80) %>%
  mutate(
    pair = map2_chr(
      as.character(feature_1), as.character(feature_2),
      ~ paste(sort(c(.x, .y)), collapse = " | ")
    )
  ) %>%
  distinct(pair, .keep_all = TRUE) %>%
  arrange(desc(abs(correlation)))

print(high_correlations, n = Inf)


# ============================================================
# STEP 29 - XI STRENGTH DECILES VS OUTCOME  (TRAINING ONLY)
# ============================================================
# Does a growing home-vs-away XI advantage change the H/D/A mix?
strength_deciles <- train_data %>%
  filter(!is.na(xi_strength_diff)) %>%
  mutate(strength_decile = ntile(xi_strength_diff, 10)) %>%
  count(strength_decile, target_result) %>%
  group_by(strength_decile) %>%
  mutate(percentage = 100 * n / sum(n)) %>%
  ungroup()

print(strength_deciles)

ggplot(strength_deciles, aes(x = strength_decile, y = percentage, colour = target_result)) +
  geom_line() +
  geom_point() +
  labs(title = "Match Outcome by XI Strength-Difference Decile (training period)",
       x = "XI Strength Difference Decile", y = "Outcome Percentage",
       colour = "Outcome") +
  theme_minimal()


# ============================================================
# STEP 30 - FINAL DESCRIPTIVE SUMMARY
# ============================================================
cat("\n=====================================\n")
cat("DESCRIPTIVE ANALYSIS COMPLETE\n")
cat("=====================================\n")

cat("Rows (all matches)    :", nrow(football), "\n")
cat("Rows (training period):", nrow(train_data), "\n")
cat("Columns (original file):", n_columns_original, "\n")
cat("Duplicate matches     :", duplicate_match_uid, "\n")
cat("Target mismatches     :", target_mismatch, "\n")

cat("\nOutcome-related plots and correlations (Steps 21, 22, 27, 28, 29)\n")
cat("use the training period only.\n")
cat("Check the generated CSV tables and RStudio plots.\n")
