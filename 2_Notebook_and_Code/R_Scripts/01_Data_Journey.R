# ============================================================
# IT3091 MACHINE LEARNING - FOOTBALL ANALYTICS PROJECT
# FILE: 01_Data_Journey.R
#
# PURPOSE
#   Describe HOW the final dataset was made and WHAT it looks like now:
#     Part A  the journey: raw files -> Stage 1 / Stage 2 -> Master -> Primary
#     Part B  rows kept and removed at each step, and why
#     Part C  how the two sources fit together over time
#     Part D  data quality by stage (player coverage, missing values)
#     Part E  current state of the final (Primary) dataset
#
#   This script only DESCRIBES the data. It does not look at how
#   features relate to the outcome (that is in 02 and 03, on the
#   training period only), so it uses all seasons.
#
# INPUT FILES (put them in the working directory, or edit the paths)
#   Stage1_PlayerAware_Football_2008_2016_Causal_v2.csv
#   Stage2_PlayerAware_Football_2016_2026_Causal_v2.csv
#   European_Football_PlayerAware_Master_2008_2026.csv
#   European_Football_PlayerAware_Primary_9Leagues_2008_2026.csv
#   Optional: the raw folders csv/stage_1 and csv/stage_2 (for raw row counts)
#
# OUTPUT: folder outputs_01_data_journey/ (CSV tables and PNG figures)
# ============================================================


# ============================================================
# STEP 0 - PACKAGES (install once if needed)
# ============================================================
# install.packages(c("tidyverse", "data.table", "scales"))
library(tidyverse)
library(data.table)
library(scales)


# ============================================================
# STEP 1 - FILE PATHS
# ============================================================
stage1_file  <- "Stage1_PlayerAware_Football_2008_2016_Causal_v2.csv"
stage2_file  <- "Stage2_PlayerAware_Football_2016_2026_Causal_v2.csv"
master_file  <- "European_Football_PlayerAware_Master_2008_2026.csv"
primary_file <- "European_Football_PlayerAware_Primary_9Leagues_2008_2026.csv"
raw_stage1_dir <- "csv/stage_1"     # optional
raw_stage2_dir <- "csv/stage_2"     # optional

out_dir <- "outputs_01_data_journey"
dir.create(out_dir, showWarnings = FALSE)

primary_leagues <- c("BE1", "ES1", "FR1", "GB1", "IT1", "L1", "NL1", "PO1", "SC1")

read_stage <- function(path) {
  read_csv(path, show_col_types = FALSE, guess_max = 60000) %>%
    mutate(date = as.Date(date), season_start = as.integer(season_start))
}
stage1  <- read_stage(stage1_file)
stage2  <- read_stage(stage2_file)
master  <- read_stage(master_file)
primary <- read_stage(primary_file)

cat("Stage 1:", dim(stage1), "| Stage 2:", dim(stage2),
    "| Master:", dim(master), "| Primary:", dim(primary), "\n")


# ============================================================
# STEP 2 - RAW FILE INVENTORY (only if the raw folders exist)
# ============================================================
# WHY? Shows the starting point of the journey: how big each raw table is
# and whether it was used. fread(select = 1) counts rows quickly.
find_raw <- function(folder, name) {
  if (!dir.exists(folder)) return(NA_character_)
  files <- list.files(folder, pattern = "\\.csv$", full.names = TRUE)
  norm  <- function(x) gsub("[^a-z0-9]", "", tolower(x))
  hit   <- files[norm(tools::file_path_sans_ext(basename(files))) == norm(name)]
  if (length(hit) == 0) NA_character_ else hit[1]
}
raw_tables <- tribble(
  ~source,   ~table,              ~used, ~role_in_project,
  "Stage 1", "Match",             "Yes", "matches, goals, starting XI player IDs and pitch coordinates",
  "Stage 1", "League",            "Yes", "league names",
  "Stage 1", "country",           "Yes", "country names",
  "Stage 1", "Team",              "Yes", "club names",
  "Stage 1", "Player Attributes", "Yes", "dated FIFA overall ratings",
  "Stage 1", "player",            "No",  "names and birth dates only",
  "Stage 1", "Team Attributes",   "No",  "team tactical ratings (not player-aware)",
  "Stage 2", "games",             "Yes", "matches, goals, formations, clubs",
  "Stage 2", "competitions",      "Yes", "league names",
  "Stage 2", "game_lineups",      "Yes", "starting XI and position text",
  "Stage 2", "player_valuations", "Yes", "dated market values",
  "Stage 2", "appearances",       "No",  "in-match statistics (would leak the result)",
  "Stage 2", "game_events",       "No",  "goals, cards, substitutions (would leak the result)",
  "Stage 2", "club_games",        "No",  "duplicate view of games",
  "Stage 2", "players",           "No",  "current player profile (not dated, so not causal)",
  "Stage 2", "transfers",         "No",  "not needed for the research questions",
  "Stage 2", "clubs",             "No",  "current club profile",
  "Stage 2", "countries",         "No",  "not needed",
  "Stage 2", "national_teams",    "No",  "not needed"
)
raw_tables$rows <- map2_dbl(raw_tables$source, raw_tables$table, function(src, tb) {
  p <- find_raw(if (src == "Stage 1") raw_stage1_dir else raw_stage2_dir, tb)
  if (is.na(p)) NA_real_ else nrow(fread(p, select = 1L, showProgress = FALSE))
})
print(raw_tables, n = Inf)
write_csv(raw_tables, file.path(out_dir, "A_raw_table_inventory.csv"))


# ============================================================
# STEP 3 - THE JOURNEY TABLE
# ============================================================
# WHY? One table that tells the whole story from source to final file.
describe_step <- function(df, step, note) {
  tibble(step = step,
         matches = nrow(df), columns = ncol(df),
         leagues = n_distinct(df$league_code),
         first_match = min(df$date), last_match = max(df$date),
         seasons = n_distinct(df$season_start),
         player_ready_pct = round(100 * mean(df$player_feature_ready == 1), 1),
         note = note)
}
journey <- bind_rows(
  describe_step(stage1,  "1. Stage 1 engineered (European Soccer SQLite)", "11 leagues, 2008/09-2015/16, FIFA rating -> percentile"),
  describe_step(stage2,  "2. Stage 2 engineered (Transfermarkt)",        "11 leagues, 2016/17-2025/26, market value -> percentile"),
  describe_step(master,  "3. Master = Stage 1 + Stage 2 (rows stacked)",  "same 62 columns, IDs prefixed S1_ / TM_"),
  describe_step(primary, "4. Primary = Master without C1 and PL1",        "final dataset used by the ML pipeline")
)
print(journey, width = Inf)
write_csv(journey, file.path(out_dir, "B_dataset_journey.csv"))

# Consistency checks on the journey (the script stops if one fails)
stopifnot(nrow(master) == nrow(stage1) + nrow(stage2))
stopifnot(!any(duplicated(master$match_uid)))
stopifnot(all(primary$match_uid %in% master$match_uid))
stopifnot(max(stage1$date) < min(stage2$date))
cat("Journey checks passed: Master = Stage 1 + Stage 2, no duplicate IDs, stages do not overlap.\n")


# ============================================================
# STEP 4 - ROWS REMOVED FROM MASTER TO PRIMARY, AND WHY
# ============================================================
removed <- master %>%
  filter(!league_code %in% primary_leagues) %>%
  count(league_code, league_name, data_source, name = "matches_removed") %>%
  arrange(league_code, data_source)
print(removed)
write_csv(removed, file.path(out_dir, "C_rows_removed.csv"))

why_removed <- master %>%
  mutate(group = if_else(league_code %in% c("C1", "PL1"), as.character(league_code), "9 primary leagues")) %>%
  group_by(group) %>%
  summarise(matches = n(),
            seasons_covered = paste(sort(unique(season_start)), collapse = " "),
            no_player_data_pct = round(100 * mean(player_feature_ready == 0), 1), .groups = "drop")
print(why_removed, width = Inf)
write_csv(why_removed, file.path(out_dir, "C_why_C1_PL1_removed.csv"))
cat(sprintf("Removed %d of %d matches (%.1f%%).\n", nrow(master) - nrow(primary), nrow(master),
            100 * (nrow(master) - nrow(primary)) / nrow(master)))


# ============================================================
# STEP 5 - HOW THE TWO SOURCES FIT TOGETHER OVER TIME
# ============================================================
season_source <- master %>%
  mutate(in_primary = if_else(league_code %in% primary_leagues, "Kept in Primary", "Removed (C1, PL1)"),
         source = if_else(str_detect(data_source, "sqlite"), "Stage 1: SQLite", "Stage 2: Transfermarkt")) %>%
  count(season_start, source, in_primary)
write_csv(season_source, file.path(out_dir, "D_matches_per_season_by_source.csv"))

p1 <- ggplot(season_source, aes(factor(season_start), n, fill = source, alpha = in_primary)) +
  geom_col() +
  scale_alpha_manual(values = c("Kept in Primary" = 1, "Removed (C1, PL1)" = 0.35), name = NULL) +
  geom_vline(xintercept = 8.5, linetype = "dashed") +
  annotate("text", x = 8.5, y = Inf, label = "source handover (summer 2016)", vjust = 1.5, hjust = -0.05, size = 3.2) +
  labs(title = "Matches per season and source",
       subtitle = "Faded bars are Switzerland and Poland, removed from the Primary dataset",
       x = "Season start", y = "Matches", fill = NULL) +
  theme_minimal() + theme(axis.text.x = element_text(angle = 45, hjust = 1), legend.position = "bottom")
ggsave(file.path(out_dir, "fig_D1_matches_per_season_by_source.png"), p1, width = 10, height = 5.5, dpi = 150)

# Do the outcome shares change at the handover? (description only)
handover <- primary %>%
  filter(season_start %in% 2012:2019) %>%
  mutate(period = if_else(season_start <= 2015, "2012-2015 (Stage 1)", "2016-2019 (Stage 2)")) %>%
  count(period, target_result) %>% group_by(period) %>% mutate(share = round(n / sum(n), 3)) %>% ungroup()
print(handover)
write_csv(handover, file.path(out_dir, "D_outcome_shares_around_handover.csv"))


# ============================================================
# STEP 6 - DATA QUALITY BY STAGE
# ============================================================
# 6a. Player-feature coverage by season and source
coverage <- master %>%
  filter(league_code %in% primary_leagues) %>%
  group_by(season_start, data_source) %>%
  summarise(matches = n(), player_ready_pct = 100 * mean(player_feature_ready == 1),
            mean_strength_coverage = mean((home_strength_coverage + away_strength_coverage) / 2, na.rm = TRUE),
            .groups = "drop")
write_csv(coverage, file.path(out_dir, "E_player_coverage_by_season.csv"))

p2 <- ggplot(coverage, aes(season_start, player_ready_pct, colour = data_source)) +
  geom_line(linewidth = 1) + geom_point() +
  scale_y_continuous(limits = c(0, 100)) +
  labs(title = "Share of matches with usable player data (player_feature_ready = 1)",
       subtitle = "Nine primary leagues", x = "Season start", y = "% of matches", colour = NULL) +
  theme_minimal() + theme(legend.position = "bottom")
ggsave(file.path(out_dir, "fig_E1_player_coverage_by_season.png"), p2, width = 9, height = 5, dpi = 150)

# 6b. Missing values by stage (same 62 columns in every file)
missing_by_stage <- bind_rows(
  stage1  %>% summarise(across(everything(), ~ mean(is.na(.x)) * 100)) %>% mutate(file = "Stage 1"),
  stage2  %>% summarise(across(everything(), ~ mean(is.na(.x)) * 100)) %>% mutate(file = "Stage 2"),
  primary %>% summarise(across(everything(), ~ mean(is.na(.x)) * 100)) %>% mutate(file = "Primary")
) %>%
  pivot_longer(-file, names_to = "column", values_to = "missing_pct") %>%
  pivot_wider(names_from = file, values_from = missing_pct) %>%
  filter(Primary > 0 | `Stage 1` > 0 | `Stage 2` > 0) %>%
  arrange(desc(Primary)) %>%
  mutate(across(where(is.numeric), ~ round(.x, 2)))
print(missing_by_stage, n = 15)
write_csv(missing_by_stage, file.path(out_dir, "E_missing_values_by_stage.csv"))

# 6c. Why values are missing (structural reasons)
season_opener <- primary %>%
  arrange(home_team_id, date) %>%
  summarise(form_missing_rows = sum(is.na(home_points_per_game_L5)),
            of_which_first_match_of_season = sum(is.na(home_points_per_game_L5) & home_form_n_L5 == 0))
print(season_opener)

missing_reasons <- tribble(
  ~feature_group, ~missing_pct_primary, ~reason,
  "Recent form (L5)", round(100 * mean(is.na(primary$home_points_per_game_L5)), 2),
    "team's first match of a season: the 5-match window restarts each season",
  "Lineup continuity", round(100 * mean(is.na(primary$home_lineup_continuity)), 2),
    "needs the previous same-season match with a complete XI of 11",
  "XI strength", round(100 * mean(is.na(primary$home_xi_strength)), 2),
    "fewer than 8 starters with a rating or value before the match",
  "Goalkeeper strength", round(100 * mean(is.na(primary$home_gk_strength)), 2),
    "goalkeeper has no rating or value, or fewer than 30 earlier goalkeepers in that league",
  "Formation", round(100 * mean(is.na(primary$home_formation)), 2),
    "formation not recorded in the source"
)
print(missing_reasons, width = Inf)
write_csv(missing_reasons, file.path(out_dir, "E_missing_value_reasons.csv"))


# ============================================================
# STEP 7 - CURRENT STATE OF THE FINAL (PRIMARY) DATASET
# ============================================================
column_groups <- tribble(
  ~group, ~n_columns, ~model_input,
  "Identity (IDs, names, dates, league, source)", 15, "only league_code",
  "Recent form (last 5 matches)", 12, "yes (FS1)",
  "Starting-XI evidence (counts, coverage, value age)", 8, "counts and coverage (FS2)",
  "Player strength (0-100) and differences", 19, "yes (FS2, FS3)",
  "Formation and lineup continuity", 4, "tested in FS4, not selected",
  "Quality flag (player_feature_ready)", 1, "no",
  "Outcome (two goal columns + target)", 3, "never; target only"
)
stopifnot(sum(column_groups$n_columns) == ncol(primary))

state <- tibble(
  item = c("Matches", "Columns", "Leagues", "Seasons", "First match", "Last match",
           "Home wins (H)", "Draws (D)", "Away wins (A)",
           "Matches with usable player data", "Columns with any missing value", "Largest missing share",
           "Duplicate match IDs", "Target matches the goals"),
  value = c(format(nrow(primary), big.mark = ","), ncol(primary), n_distinct(primary$league_code),
            n_distinct(primary$season_start), as.character(min(primary$date)), as.character(max(primary$date)),
            sprintf("%s (%.1f%%)", format(sum(primary$target_result == "H"), big.mark = ","), 100 * mean(primary$target_result == "H")),
            sprintf("%s (%.1f%%)", format(sum(primary$target_result == "D"), big.mark = ","), 100 * mean(primary$target_result == "D")),
            sprintf("%s (%.1f%%)", format(sum(primary$target_result == "A"), big.mark = ","), 100 * mean(primary$target_result == "A")),
            sprintf("%.1f%%", 100 * mean(primary$player_feature_ready == 1)),
            sum(colSums(is.na(primary)) > 0),
            sprintf("%.1f%% (%s)", 100 * max(colMeans(is.na(primary))), names(which.max(colMeans(is.na(primary))))),
            sum(duplicated(primary$match_uid)),
            as.character(all(with(primary, ifelse(audit_home_goals > audit_away_goals, "H",
                                           ifelse(audit_home_goals < audit_away_goals, "A", "D"))) == primary$target_result)))
)
print(state, n = Inf)
write_csv(state, file.path(out_dir, "F_current_state_summary.csv"))
write_csv(column_groups, file.path(out_dir, "F_column_groups.csv"))

split_table <- primary %>%
  mutate(split = case_when(season_start <= 2022 ~ "Train (2008/09-2022/23)",
                           season_start == 2023 ~ "Validation (2023/24)",
                           TRUE ~ "Test (2024/25-2025/26)")) %>%
  count(split, target_result) %>%
  pivot_wider(names_from = target_result, values_from = n) %>%
  mutate(total = H + D + A)
print(split_table)
write_csv(split_table, file.path(out_dir, "F_matches_per_split.csv"))

p3 <- primary %>% count(league_code, data_source) %>%
  ggplot(aes(reorder(league_code, n, sum), n, fill = data_source)) + geom_col() + coord_flip() +
  labs(title = "Final dataset: matches per league and source", x = NULL, y = "Matches", fill = NULL) +
  theme_minimal() + theme(legend.position = "bottom")
ggsave(file.path(out_dir, "fig_F1_matches_per_league.png"), p3, width = 8, height = 5, dpi = 150)

cat("\nDATA JOURNEY COMPLETE. Tables and figures saved in", out_dir, "\n")
