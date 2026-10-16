# ============================================================
# IT3091 MACHINE LEARNING - FOOTBALL ANALYTICS PROJECT
# FILE: 04_Time_and_Model_Inference.R
#
# PURPOSE (adds three inference questions to 03_Statistical_Inference.R)
#   Part A  Is the data time dependent?
#           - Do outcome shares change across seasons and between the two sources?
#           - Did home advantage drop in the COVID season (empty stadiums)?
#           - Is there a long-term trend in home wins?
#           - Does the link between XI strength and the result change over time?
#   Part B  Multicollinearity: which features overlap (VIF)?
#   Part C  Does player information improve on team form? (formal test of RQ2)
#           Nested multinomial logistic regressions compared with
#           likelihood-ratio tests.
#
# IMPORTANT: only TRAINING matches (season_start <= 2022) are used,
# exactly as in scripts 02 and 03. Validation and test stay untouched.
#
# OUTPUT: folder outputs_04_time_and_model/
# ============================================================


# ============================================================
# STEP 0 - PACKAGES (install once if needed)
# ============================================================
# install.packages(c("tidyverse", "car", "nnet"))
library(tidyverse)
library(car)    # vif()
library(nnet)   # multinom()


# ============================================================
# STEP 1 - LOAD THE FINAL DATASET AND KEEP THE TRAINING PERIOD
# ============================================================
file_path <- "European_Football_PlayerAware_Primary_9Leagues_2008_2026.csv"
out_dir   <- "outputs_04_time_and_model"
dir.create(out_dir, showWarnings = FALSE)

football <- read_csv(file_path, show_col_types = FALSE, guess_max = 60000) %>%
  mutate(date          = as.Date(date),
         season_start  = as.integer(season_start),
         target_result = factor(target_result, levels = c("H", "D", "A")),
         league_code   = factor(league_code),
         source        = if_else(str_detect(data_source, "sqlite"), "Stage 1 (2008-2015)", "Stage 2 (2016-2022)"))

train <- football %>% filter(season_start <= 2022)
cat("Training matches:", nrow(train), "| seasons", min(train$season_start), "to", max(train$season_start), "\n")

tests <- list()   # every test result is collected here for one summary table at the end
add_test <- function(part, question, test, statistic, df, p, effect, n) {
  tests[[length(tests) + 1]] <<- tibble(part = part, question = question, test = test,
                                         statistic = statistic, df = as.character(df), p_value = p,
                                         effect_size = effect, n = n)
}
cramers_v <- function(tab) {
  chi <- suppressWarnings(chisq.test(tab, correct = FALSE))$statistic
  as.numeric(sqrt(chi / (sum(tab) * (min(dim(tab)) - 1))))
}


# ============================================================
# PART A - TIME DEPENDENCE
# ============================================================

# ------------------------------------------------------------
# STEP A1 - Outcome shares by season (chi-square test of independence)
# ------------------------------------------------------------
# H0: the H/D/A shares are the same in every training season.
season_table <- table(train$season_start, train$target_result)
season_shares <- round(prop.table(season_table, 1), 3)
print(season_shares)
write.csv(as.data.frame.matrix(season_shares), file.path(out_dir, "A1_outcome_shares_by_season.csv"))
a1 <- chisq.test(season_table)
add_test("A Time", "Do outcome shares differ between seasons?", "Chi-square (season x outcome)",
         unname(a1$statistic), unname(a1$parameter), a1$p.value, sprintf("Cramer's V = %.3f", cramers_v(season_table)), sum(season_table))

p_season <- as.data.frame(prop.table(table(football$season_start, football$target_result), 1)) %>%
  rename(season = Var1, outcome = Var2, share = Freq) %>%
  mutate(season = as.integer(as.character(season)),
         split = case_when(season <= 2022 ~ "train", season == 2023 ~ "validation", TRUE ~ "test")) %>%
  ggplot(aes(season, share, colour = outcome)) +
  annotate("rect", xmin = 2019.5, xmax = 2020.5, ymin = -Inf, ymax = Inf, alpha = 0.12) +
  annotate("text", x = 2020, y = 0.52, label = "COVID season\n(empty stadiums)", size = 3) +
  geom_vline(xintercept = c(2022.5, 2023.5), linetype = "dashed", colour = "grey50") +
  geom_line(linewidth = 1) + geom_point() +
  scale_y_continuous(labels = scales::percent, limits = c(0.15, 0.55)) +
  labs(title = "Outcome shares by season (all seasons, description only)",
       subtitle = "Dashed lines: train | validation | test boundaries", x = "Season start", y = "Share of matches", colour = NULL) +
  theme_minimal() + theme(legend.position = "bottom")
ggsave(file.path(out_dir, "fig_A1_outcome_shares_by_season.png"), p_season, width = 9, height = 5, dpi = 150)

# ------------------------------------------------------------
# STEP A2 - Do the two data sources have the same outcome shares?
# ------------------------------------------------------------
source_table <- table(train$source, train$target_result)
print(round(prop.table(source_table, 1), 3))
a2 <- chisq.test(source_table)
add_test("A Time", "Do outcome shares differ between Stage 1 and Stage 2?", "Chi-square (source x outcome)",
         unname(a2$statistic), unname(a2$parameter), a2$p.value, sprintf("Cramer's V = %.3f", cramers_v(source_table)), sum(source_table))

# ------------------------------------------------------------
# STEP A3 - Home advantage in the COVID season (2020/21)
# ------------------------------------------------------------
# H0: the home-win share in 2020/21 equals the share in the other training seasons.
covid <- train %>% mutate(covid = season_start == 2020, home_win = target_result == "H")
covid_counts <- covid %>% group_by(covid) %>% summarise(home_wins = sum(home_win), matches = n(), share = mean(home_win))
print(covid_counts)
a3 <- prop.test(x = rev(covid_counts$home_wins), n = rev(covid_counts$matches))
add_test("A Time", "Was the home-win share lower in the COVID season 2020/21?", "Two-proportion test",
         unname(a3$statistic), unname(a3$parameter), a3$p.value,
         sprintf("%.1f%% vs %.1f%% (difference %.1f points)", 100 * covid_counts$share[2], 100 * covid_counts$share[1],
                 100 * (covid_counts$share[2] - covid_counts$share[1])), nrow(covid))

# ------------------------------------------------------------
# STEP A4 - Long-term trend in home wins (excluding the COVID season)
# ------------------------------------------------------------
trend_model <- glm(home_win ~ I(season_start - 2008), family = binomial, data = filter(covid, !covid))
trend <- summary(trend_model)$coefficients[2, ]
print(summary(trend_model)$coefficients)
add_test("A Time", "Is there a long-term trend in home wins (COVID season excluded)?", "Logistic regression slope (Wald z)",
         unname(trend["z value"]), 1, unname(trend["Pr(>|z|)"]),
         sprintf("odds ratio per season = %.4f", exp(trend["Estimate"])), sum(!covid$covid))

# ------------------------------------------------------------
# STEP A5 - Does the XI-strength effect change over time?
# ------------------------------------------------------------
# (a) Effect size of xi_strength_diff on the outcome in every season
eps_by_season <- train %>% filter(!is.na(xi_strength_diff)) %>%
  group_by(season_start) %>%
  summarise(n = n(),
            kw_H = kruskal.test(xi_strength_diff ~ target_result)$statistic,
            epsilon_sq = kw_H / (n - 1), .groups = "drop")
print(eps_by_season)
write_csv(eps_by_season, file.path(out_dir, "A5_xi_effect_by_season.csv"))
p_eps <- ggplot(eps_by_season, aes(season_start, epsilon_sq)) +
  geom_col(fill = "steelblue") +
  geom_hline(yintercept = mean(eps_by_season$epsilon_sq), linetype = "dashed") +
  labs(title = "How strongly XI strength difference separates H / D / A, by season",
       subtitle = "Kruskal-Wallis epsilon-squared (dashed line = average); training seasons only",
       x = "Season start", y = "epsilon-squared") + theme_minimal()
ggsave(file.path(out_dir, "fig_A5_xi_effect_by_season.png"), p_eps, width = 9, height = 4.5, dpi = 150)

# (b) Formal test: does the slope of xi_strength_diff differ between the two sources/eras?
#     Likelihood-ratio test of the interaction xi_strength_diff x source.
era_data <- train %>% filter(!is.na(xi_strength_diff)) %>%
  mutate(target = relevel(target_result, ref = "D"), xi_z = as.numeric(scale(xi_strength_diff)))
m_main <- multinom(target ~ xi_z + source, data = era_data, trace = FALSE)
m_int  <- multinom(target ~ xi_z * source, data = era_data, trace = FALSE)
lr_era <- anova(m_main, m_int)
print(lr_era)
add_test("A Time", "Does the XI-strength effect differ between Stage 1 and Stage 2?", "LR test (xi x source interaction)",
         lr_era$`LR stat.`[2], lr_era$`   Df`[2], lr_era$`Pr(Chi)`[2],
         "see A5_xi_effect_by_season.csv", nrow(era_data))


# ============================================================
# PART B - MULTICOLLINEARITY
# ============================================================

# ------------------------------------------------------------
# STEP B1 - Exact identities (columns that are sums or differences of others)
# ------------------------------------------------------------
identities <- tibble(
  identity = c("home_goal_diff_avg_L5 = home_goals_for_avg_L5 - home_goals_against_avg_L5",
               "xi_strength_diff = home_xi_strength - away_xi_strength",
               "home_attack_vs_away_defence = home_attack_strength - away_defence_strength",
               "home_top3_gap = home_top3_strength - home_xi_strength"),
  max_abs_error = c(
    max(abs(train$home_goal_diff_avg_L5 - (train$home_goals_for_avg_L5 - train$home_goals_against_avg_L5)), na.rm = TRUE),
    max(abs(train$xi_strength_diff - (train$home_xi_strength - train$away_xi_strength)), na.rm = TRUE),
    max(abs(train$home_attack_vs_away_defence - (train$home_attack_strength - train$away_defence_strength)), na.rm = TRUE),
    max(abs(train$home_top3_gap - (train$home_top3_strength - train$home_xi_strength)), na.rm = TRUE)))
print(identities)
write_csv(identities, file.path(out_dir, "B1_exact_identities.csv"))

# ------------------------------------------------------------
# STEP B2 - Variance Inflation Factor (VIF)
# ------------------------------------------------------------
# VIF_j = 1 / (1 - R2_j), where R2_j comes from regressing feature j on all
# other features. It depends only on the predictors, so any numeric response
# can be used to fit the helper model. Rule of thumb: VIF > 10 = strong overlap.
# The exact identities from B1 are left out (their VIF is infinite).
vif_features <- c("home_form_n_L5", "away_form_n_L5",
                  "home_points_per_game_L5", "away_points_per_game_L5",
                  "home_win_rate_L5", "away_win_rate_L5",
                  "home_goals_for_avg_L5", "away_goals_for_avg_L5",
                  "home_goals_against_avg_L5", "away_goals_against_avg_L5",
                  "home_starter_count", "away_starter_count",
                  "home_strength_coverage", "away_strength_coverage",
                  "home_xi_strength", "away_xi_strength",
                  "home_top3_strength", "away_top3_strength",
                  "home_attack_strength", "away_attack_strength",
                  "home_midfield_strength", "away_midfield_strength",
                  "home_defence_strength", "away_defence_strength",
                  "home_gk_strength", "away_gk_strength")
vif_data <- train %>% select(audit_home_goals, all_of(vif_features)) %>% drop_na()
vif_model <- lm(audit_home_goals ~ ., data = vif_data)
vif_table <- tibble(feature = names(vif(vif_model)), VIF = round(unname(vif(vif_model)), 1)) %>%
  arrange(desc(VIF)) %>%
  mutate(level = case_when(VIF > 10 ~ "high (> 10)", VIF > 5 ~ "moderate (5-10)", TRUE ~ "low (< 5)"))
print(vif_table, n = Inf)
cat("Rows used for VIF (complete cases):", nrow(vif_data), "\n")
write_csv(vif_table, file.path(out_dir, "B2_vif_table.csv"))

p_vif <- ggplot(vif_table, aes(reorder(feature, VIF), VIF, fill = level)) +
  geom_col() + coord_flip() + geom_hline(yintercept = c(5, 10), linetype = "dashed") +
  scale_y_log10() +
  labs(title = "Variance Inflation Factor of the model features (training period)",
       subtitle = "Log scale; dashed lines at 5 and 10", x = NULL, y = "VIF", fill = NULL) +
  theme_minimal() + theme(legend.position = "bottom")
ggsave(file.path(out_dir, "fig_B2_vif.png"), p_vif, width = 8, height = 7, dpi = 150)


# ============================================================
# PART C - DOES PLAYER INFORMATION ADD TO TEAM FORM? (RQ2, inferential)
# ============================================================
# Nested multinomial logistic regressions on the SAME matches.
# Each model adds one block; a likelihood-ratio test asks whether the block
# explains the outcome better than the smaller model. Predictors are
# standardised (1 unit = 1 standard deviation) so odds ratios are comparable.
#   M0  league only (baseline)
#   M1  + recent form          (points-per-game and goal-difference, home minus away)
#   M2  + starting-XI quality  (xi_strength_diff)
#   M3  + positional matchups  (attack vs defence, midfield, goalkeeper differences)
#   M4  + lineup continuity    (home minus away)
model_data <- train %>%
  transmute(target = relevel(target_result, ref = "D"), league_code,
            ppg_diff = home_points_per_game_L5 - away_points_per_game_L5,
            gd_diff  = home_goal_diff_avg_L5 - away_goal_diff_avg_L5,
            xi_strength_diff, home_attack_vs_away_defence, away_attack_vs_home_defence,
            midfield_strength_diff, gk_strength_diff,
            continuity_diff = home_lineup_continuity - away_lineup_continuity) %>%
  drop_na() %>%
  mutate(across(where(is.numeric), ~ as.numeric(scale(.x))))
cat("Matches used in the nested models (complete cases):", nrow(model_data), "\n")

f0 <- target ~ league_code
f1 <- update(f0, . ~ . + ppg_diff + gd_diff)
f2 <- update(f1, . ~ . + xi_strength_diff)
f3 <- update(f2, . ~ . + home_attack_vs_away_defence + away_attack_vs_home_defence + midfield_strength_diff + gk_strength_diff)
f4 <- update(f3, . ~ . + continuity_diff)
models <- lapply(list(f0, f1, f2, f3, f4), function(f) multinom(f, data = model_data, trace = FALSE, maxit = 500, Hess = TRUE))
names(models) <- c("M0 league", "M1 + form", "M2 + XI quality", "M3 + positional", "M4 + continuity")

ll_null <- as.numeric(logLik(multinom(target ~ 1, data = model_data, trace = FALSE)))
fit_table <- tibble(model = names(models),
                    parameters = sapply(models, function(m) attr(logLik(m), "df")),
                    log_likelihood = sapply(models, function(m) as.numeric(logLik(m))),
                    AIC = sapply(models, AIC),
                    mcfadden_R2 = 1 - log_likelihood / ll_null)
lr <- do.call(anova, unname(models))
fit_table$LR_stat <- c(NA, lr$`LR stat.`[-1])
fit_table$df_added <- c(NA, lr$`   Df`[-1])
fit_table$p_value <- c(NA, lr$`Pr(Chi)`[-1])
print(fit_table %>% mutate(across(where(is.numeric), ~ signif(.x, 4))), width = Inf)
write_csv(fit_table, file.path(out_dir, "C1_nested_model_comparison.csv"))

for (k in 2:5) {
  add_test("C Player info (RQ2)", paste("Does", sub("^M[0-9] \\+ ", "", names(models)[k]), "improve on the previous model?"),
           paste("LR test", names(models)[k - 1], "vs", names(models)[k]),
           fit_table$LR_stat[k], fit_table$df_added[k], fit_table$p_value[k],
           sprintf("McFadden R2 %.4f -> %.4f", fit_table$mcfadden_R2[k - 1], fit_table$mcfadden_R2[k]), nrow(model_data))
}

# Odds ratios from M2: effect of 1 SD more XI-strength advantage for the home side
or_table <- {
  co <- summary(models[[3]])$coefficients; se <- summary(models[[3]])$standard.errors
  tibble(outcome_vs_draw = rownames(co),
         term = "xi_strength_diff (per 1 SD)",
         odds_ratio = exp(co[, "xi_strength_diff"]),
         ci_low = exp(co[, "xi_strength_diff"] - 1.96 * se[, "xi_strength_diff"]),
         ci_high = exp(co[, "xi_strength_diff"] + 1.96 * se[, "xi_strength_diff"]))
}
print(or_table)
write_csv(or_table, file.path(out_dir, "C2_odds_ratios_xi_strength.csv"))


# ============================================================
# SUMMARY TABLE WITH HOLM CORRECTION
# ============================================================
summary_table <- bind_rows(tests) %>%
  mutate(p_holm = p.adjust(p_value, method = "holm"),
         p_value_txt = ifelse(p_value < 2.2e-16, "< 2.2e-16", formatC(p_value, format = "g", digits = 3)),
         p_holm_txt  = ifelse(p_holm < 2.2e-16, "< 2.2e-16", formatC(p_holm, format = "g", digits = 3)),
         decision = ifelse(p_holm < 0.05, "Reject H0 (significant)", "Do not reject H0"),
         statistic = round(statistic, 2))
print(summary_table %>% select(question, test, statistic, df, p_holm_txt, effect_size, decision), width = Inf)
write_csv(summary_table, file.path(out_dir, "summary_time_and_model_inference.csv"))


# ============================================================
# HOW TO READ THE RESULTS
# ============================================================
# * With about 40,000 matches, even tiny differences become "significant".
#   Always read the effect size (Cramer's V, percentage points, R2) next to p.
# * Part A tells you whether time matters. A significant but small Cramer's V
#   means outcome shares move a little between seasons; that supports the
#   chronological train / validation / test split rather than a random split.
# * Part B: high VIF does not hurt a Random Forest's predictions, but it means
#   single-feature importance and linear coefficients cannot be read one by one.
#   That is why the ML notebook reports importance by feature GROUP.
# * Part C answers RQ2 with a statistical test instead of only a score
#   comparison: if adding XI quality (M1 -> M2) gives a significant LR test
#   and a clear R2 gain, player information adds to team form.
cat("\nSCRIPT 04 COMPLETE. Results saved in", out_dir, "\n")
