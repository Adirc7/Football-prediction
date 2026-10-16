# ============================================================
# IT3091 MACHINE LEARNING - FOOTBALL ANALYTICS PROJECT
# FILE: 03_Statistical_Inference.R   (corrected version)
#
# PURPOSE:
# Inferential statistical tests on the final engineered dataset.
#
# IMPORTANT:
# Only TRAINING PERIOD matches (season_start <= 2022) are used.
# Validation 2023 and test 2024-25 remain untouched.
#
# WHAT CHANGED COMPARED WITH THE FIRST VERSION
# --------------------------------------------
# 1. The Kruskal-Wallis effect size now uses the standard
#    epsilon-squared formula  H / (n - 1).  (The first version used
#    (H - k + 1) / (n - k), which is the eta-squared-H variant.)
# 2. cor.test() no longer receives the unused `use` argument.
# 3. A Holm correction is applied across the 10 tests in the summary
#    table, and the significance label is based on the ADJUSTED p.
# 4. Very small p-values are reported as "< 2.2e-16" instead of 0,
#    and the sample size of each test is added to the summary table.
# 5. Effect sizes are saved for ALL the Kruskal-Wallis tests, not only
#    two of them.
# 6. The summary CSV is written after the adjusted p-values exist.
# ============================================================


# ============================================================
# STEP 0 - INSTALL REQUIRED PACKAGES (run once if needed)
# ============================================================
# install.packages("tidyverse")
# install.packages("car")


# ============================================================
# STEP 1 - LOAD LIBRARIES
# ============================================================
library(tidyverse)
library(car)


# ============================================================
# STEP 2 - LOAD FINAL DATASET
# ============================================================
file_path <- "European_Football_PlayerAware_Primary_9Leagues_2008_2026.csv"

football <- read_csv(file_path, show_col_types = FALSE) %>%
  select(-any_of("index")) %>%
  mutate(
    date          = as.Date(date),
    season_start  = as.integer(season_start),
    target_result = factor(target_result, levels = c("H", "D", "A")),
    league_code   = factor(league_code)
  )


# ============================================================
# STEP 3 - CREATE TRAINING-ONLY ANALYSIS DATASET
# ============================================================
# WHY? We must not inspect validation/test target relationships and
# then use those findings to design the model.
train_data <- football %>% filter(season_start <= 2022)

cat("Training rows:", nrow(train_data), "\n")


# ============================================================
# STEP 4 - CREATE ADDITIONAL DIFFERENCE VARIABLES
# ============================================================
# These compare the two competing teams directly.
train_data <- train_data %>%
  mutate(
    ppg_diff        = home_points_per_game_L5 - away_points_per_game_L5,
    win_rate_diff   = home_win_rate_L5        - away_win_rate_L5,
    form_goal_diff  = home_goal_diff_avg_L5   - away_goal_diff_avg_L5,
    continuity_diff = home_lineup_continuity  - away_lineup_continuity
  )


# ============================================================
# STEP 5 - TARGET PROPORTIONS + 95% CONFIDENCE INTERVALS
# ============================================================
# prop.test(correct = FALSE) gives the Wilson score interval.
total_matches <- nrow(train_data)

target_confidence_intervals <- train_data %>%
  count(target_result) %>%
  rowwise() %>%
  mutate(
    proportion = n / total_matches,
    ci_lower   = prop.test(n, total_matches, correct = FALSE)$conf.int[1],
    ci_upper   = prop.test(n, total_matches, correct = FALSE)$conf.int[2]
  ) %>%
  ungroup()

print(target_confidence_intervals)


# ============================================================
# STEP 6 - DESCRIPTIVE STATISTICS BY MATCH OUTCOME
# ============================================================
xi_group_summary <- train_data %>%
  group_by(target_result) %>%
  summarise(
    sample_size      = sum(!is.na(xi_strength_diff)),
    mean_xi_diff     = mean(xi_strength_diff, na.rm = TRUE),
    median_xi_diff   = median(xi_strength_diff, na.rm = TRUE),
    variance_xi_diff = var(xi_strength_diff, na.rm = TRUE),
    sd_xi_diff       = sd(xi_strength_diff, na.rm = TRUE),
    IQR_xi_diff      = IQR(xi_strength_diff, na.rm = TRUE),
    .groups = "drop"
  )

print(xi_group_summary)


# ============================================================
# STEP 7 - DISTRIBUTION VISUALISATION
# ============================================================
ggplot(train_data, aes(x = xi_strength_diff, colour = target_result)) +
  geom_density(na.rm = TRUE) +
  labs(title = "XI Strength Difference Distribution by Outcome (training period)",
       x = "XI Strength Difference", y = "Density", colour = "Outcome") +
  theme_minimal()


# ============================================================
# STEP 8 - Q-Q PLOT
# ============================================================
# With 40,000+ observations Shapiro-Wilk is not informative: even tiny
# departures from normality become "significant". Inspect a Q-Q plot.
ggplot(train_data, aes(sample = xi_strength_diff)) +
  stat_qq(na.rm = TRUE) +
  stat_qq_line(na.rm = TRUE) +
  labs(title = "Q-Q Plot of XI Strength Difference (training period)") +
  theme_minimal()


# ============================================================
# STEP 9 - LEVENE TEST FOR EQUALITY OF VARIANCES
# ============================================================
# H0: variance_H = variance_D = variance_A
# H1: at least one variance differs.
# (car::leveneTest centres on the median by default.)
levene_xi <- leveneTest(xi_strength_diff ~ target_result, data = train_data)

print(levene_xi)

# Interpretation:
#   p >= 0.05 -> insufficient evidence that variances differ
#   p <  0.05 -> evidence that at least one variance differs


# ============================================================
# STEP 10 - WELCH ONE-WAY ANOVA
# ============================================================
# Preferred over the standard ANOVA when group variances differ.
# H0: mean XI-strength difference is equal across H/D/A.
welch_xi <- oneway.test(xi_strength_diff ~ target_result,
                        data = train_data, var.equal = FALSE)

print(welch_xi)


# ============================================================
# STEP 11 - KRUSKAL-WALLIS TEST
# ============================================================
# Non-parametric alternative; does not need normal residuals.
# H0: the XI-strength-difference distributions are the same across H/D/A.
kw_xi <- kruskal.test(xi_strength_diff ~ target_result, data = train_data)

print(kw_xi)


# ============================================================
# STEP 12 - KRUSKAL-WALLIS EFFECT SIZE (epsilon-squared)
# ============================================================
# With a very large dataset even a small difference gives an extremely
# small p-value, so an effect size is reported as well.
#
#   epsilon^2 = H / (n - 1)        (n = observations actually tested)
#
# Rough guide: < 0.01 negligible, 0.01-0.08 small, 0.08-0.26 moderate,
# >= 0.26 large (Tomczak & Tomczak, 2014).
kruskal_epsilon_squared <- function(test, data, group_column) {
  H <- as.numeric(test$statistic)
  n <- nrow(data)
  max(0, H / (n - 1))
}

# Helper: rows that actually enter a test on `variable`
complete_for <- function(variable) {
  train_data %>% filter(!is.na(.data[[variable]]), !is.na(target_result))
}

xi_effect_size <- kruskal_epsilon_squared(kw_xi, complete_for("xi_strength_diff"),
                                          "target_result")

cat("\nXI Strength Difference Epsilon Squared:", xi_effect_size, "\n")


# ============================================================
# STEP 13 - POST-HOC WILCOXON TESTS
# ============================================================
# Kruskal-Wallis only says "at least one group differs". Compare
# H vs D, H vs A and D vs A; Holm controls the multiple-testing error.
pairwise_xi <- pairwise.wilcox.test(
  train_data$xi_strength_diff,
  train_data$target_result,
  p.adjust.method = "holm",
  exact = FALSE
)

print(pairwise_xi)


# ============================================================
# STEP 14 - RECENT FORM (PPG DIFFERENCE) VS OUTCOME
# ============================================================
kw_ppg <- kruskal.test(ppg_diff ~ target_result, data = train_data)

print(kw_ppg)

ppg_effect_size <- kruskal_epsilon_squared(kw_ppg, complete_for("ppg_diff"),
                                           "target_result")

cat("PPG Difference Effect Size:", ppg_effect_size, "\n")


# ============================================================
# STEP 15 - RECENT WIN-RATE DIFFERENCE VS OUTCOME
# ============================================================
kw_win_rate <- kruskal.test(win_rate_diff ~ target_result, data = train_data)

print(kw_win_rate)


# ============================================================
# STEP 16 - RECENT GOAL-DIFFERENCE FORM VS OUTCOME
# ============================================================
kw_form_goal_diff <- kruskal.test(form_goal_diff ~ target_result, data = train_data)

print(kw_form_goal_diff)


# ============================================================
# STEP 17 - HOME ATTACK VS AWAY DEFENCE
# ============================================================
kw_attack_defence <- kruskal.test(home_attack_vs_away_defence ~ target_result,
                                  data = train_data)

print(kw_attack_defence)


# ============================================================
# STEP 18 - MIDFIELD DIFFERENCE VS OUTCOME
# ============================================================
kw_midfield <- kruskal.test(midfield_strength_diff ~ target_result, data = train_data)

print(kw_midfield)


# ============================================================
# STEP 19 - GOALKEEPER DIFFERENCE VS OUTCOME
# ============================================================
kw_goalkeeper <- kruskal.test(gk_strength_diff ~ target_result, data = train_data)

print(kw_goalkeeper)


# ============================================================
# STEP 20 - LINEUP CONTINUITY DIFFERENCE VS OUTCOME
# ============================================================
kw_continuity <- kruskal.test(continuity_diff ~ target_result, data = train_data)

print(kw_continuity)


# ============================================================
# STEP 21 - CHI-SQUARE TEST: LEAGUE VS MATCH OUTCOME
# ============================================================
# H0: league and H/D/A result are independent.
# H1: they are statistically associated.
league_outcome_table <- table(train_data$league_code, train_data$target_result)

print(league_outcome_table)

chi_league <- chisq.test(league_outcome_table)

print(chi_league)

# Expected frequencies should generally be >= 5.
print(chi_league$expected)


# ============================================================
# STEP 22 - CRAMER'S V EFFECT SIZE
# ============================================================
# The chi-square p-value says whether an association exists; Cramer's V
# says how strong it is.
cramers_v <- function(chi_test, contingency_table) {
  chi_square <- as.numeric(chi_test$statistic)
  n          <- sum(contingency_table)
  sqrt(chi_square / (n * min(nrow(contingency_table) - 1, ncol(contingency_table) - 1)))
}

league_cramers_v <- cramers_v(chi_league, league_outcome_table)

cat("\nCramer's V - League vs Outcome:", league_cramers_v, "\n")


# ============================================================
# STEP 23 - PEARSON CORRELATION TEST
# ============================================================
# XI strength and top-3 strength are related constructions, so a high
# correlation is expected. cor.test() drops incomplete pairs itself.
cor_xi_top3 <- cor.test(train_data$home_xi_strength,
                        train_data$home_top3_strength,
                        method = "pearson")

print(cor_xi_top3)


# ============================================================
# STEP 24 - SPEARMAN CORRELATION TEST
# ============================================================
# Useful when the relationship is monotonic but not necessarily linear.
cor_xi_attack_spearman <- cor.test(train_data$home_xi_strength,
                                   train_data$home_attack_strength,
                                   method = "spearman",
                                   exact = FALSE)

print(cor_xi_attack_spearman)


# ============================================================
# STEP 25 - BUILD ONE FINAL INFERENCE SUMMARY TABLE
# ============================================================
inference_summary <- tibble(
  analysis = c(
    "Welch ANOVA - XI Strength Difference",
    "Kruskal-Wallis - XI Strength Difference",
    "Kruskal-Wallis - PPG Difference",
    "Kruskal-Wallis - Win Rate Difference",
    "Kruskal-Wallis - Form Goal Difference",
    "Kruskal-Wallis - Home Attack vs Away Defence",
    "Kruskal-Wallis - Midfield Strength Difference",
    "Kruskal-Wallis - Goalkeeper Strength Difference",
    "Kruskal-Wallis - Lineup Continuity Difference",
    "Chi-square - League vs Match Outcome"
  ),
  n = c(
    nrow(complete_for("xi_strength_diff")),
    nrow(complete_for("xi_strength_diff")),
    nrow(complete_for("ppg_diff")),
    nrow(complete_for("win_rate_diff")),
    nrow(complete_for("form_goal_diff")),
    nrow(complete_for("home_attack_vs_away_defence")),
    nrow(complete_for("midfield_strength_diff")),
    nrow(complete_for("gk_strength_diff")),
    nrow(complete_for("continuity_diff")),
    sum(league_outcome_table)
  ),
  statistic = c(
    as.numeric(welch_xi$statistic),
    as.numeric(kw_xi$statistic),
    as.numeric(kw_ppg$statistic),
    as.numeric(kw_win_rate$statistic),
    as.numeric(kw_form_goal_diff$statistic),
    as.numeric(kw_attack_defence$statistic),
    as.numeric(kw_midfield$statistic),
    as.numeric(kw_goalkeeper$statistic),
    as.numeric(kw_continuity$statistic),
    as.numeric(chi_league$statistic)
  ),
  p_value = c(
    welch_xi$p.value,
    kw_xi$p.value,
    kw_ppg$p.value,
    kw_win_rate$p.value,
    kw_form_goal_diff$p.value,
    kw_attack_defence$p.value,
    kw_midfield$p.value,
    kw_goalkeeper$p.value,
    kw_continuity$p.value,
    chi_league$p.value
  )
)


# ============================================================
# STEP 26 - MULTIPLE-TESTING CORRECTION AND SIGNIFICANCE LABEL
# ============================================================
# Ten tests are reported together, so p-values are adjusted with Holm.
# The label uses the ADJUSTED p-value.
format_p <- function(p) {
  ifelse(p < 2.2e-16, "< 2.2e-16", formatC(p, format = "e", digits = 2))
}

inference_summary <- inference_summary %>%
  mutate(
    p_adjusted_holm = p.adjust(p_value, method = "holm"),
    p_value_text    = format_p(p_value),
    p_adjusted_text = format_p(p_adjusted_holm),
    significance = case_when(
      p_adjusted_holm < 0.001 ~ "Highly statistically significant",
      p_adjusted_holm < 0.05  ~ "Statistically significant",
      TRUE                    ~ "Not statistically significant"
    )
  )

print(inference_summary, n = Inf, width = Inf)

write_csv(inference_summary, "statistical_inference_summary.csv")


# ============================================================
# STEP 27 - SAVE EFFECT-SIZE SUMMARY
# ============================================================
effect_size_summary <- tibble(
  analysis = c(
    "XI Strength Difference vs Outcome",
    "PPG Difference vs Outcome",
    "Win Rate Difference vs Outcome",
    "Form Goal Difference vs Outcome",
    "Home Attack vs Away Defence vs Outcome",
    "Midfield Strength Difference vs Outcome",
    "Goalkeeper Strength Difference vs Outcome",
    "Lineup Continuity Difference vs Outcome",
    "League vs Outcome"
  ),
  effect_size_type = c(rep("Kruskal-Wallis epsilon squared", 8), "Cramer's V"),
  effect_size = c(
    xi_effect_size,
    ppg_effect_size,
    kruskal_epsilon_squared(kw_win_rate,       complete_for("win_rate_diff"),               "target_result"),
    kruskal_epsilon_squared(kw_form_goal_diff, complete_for("form_goal_diff"),              "target_result"),
    kruskal_epsilon_squared(kw_attack_defence, complete_for("home_attack_vs_away_defence"), "target_result"),
    kruskal_epsilon_squared(kw_midfield,       complete_for("midfield_strength_diff"),      "target_result"),
    kruskal_epsilon_squared(kw_goalkeeper,     complete_for("gk_strength_diff"),            "target_result"),
    kruskal_epsilon_squared(kw_continuity,     complete_for("continuity_diff"),             "target_result"),
    league_cramers_v
  )
)

print(effect_size_summary, n = Inf)

write_csv(effect_size_summary, "statistical_effect_sizes.csv")


# ============================================================
# STEP 28 - IMPORTANT INTERPRETATION RULE
# ============================================================
cat("\n==========================================\n")
cat("STATISTICAL INFERENCE COMPLETE\n")
cat("==========================================\n")

cat("\nImportant:\n")
cat("A statistically significant result does NOT prove causation.\n")
cat("If XI strength differs significantly among H/D/A, we say XI strength\n")
cat("is ASSOCIATED with match outcome, not that a stronger XI causes the result.\n")
cat("\nBecause this dataset is large, p-values become extremely small.\n")
cat("Always discuss the effect sizes and the actual group differences too.\n")
cat("Matches are not fully independent (the same teams recur over time),\n")
cat("so treat the p-values as descriptive evidence, not exact error rates.\n")
