#!/usr/bin/env bash
# Model-age retrain gate (single source of truth).
#
# Retrain only when model_assets/level_q50.pkl has not been committed in the
# last >= 7 days. This lets the daily pipeline regenerate the forecast every
# day while retraining the models weekly.
#
# Mirrors the gate logic in .github/workflows/forecast_pipeline.yml (shadow);
# the shadow keeps its inline copy for now and can adopt this script at cutover.
#
# Requires the workflow checkout to use `fetch-depth: 0` so `git log` can see
# the history needed to find the last retrain commit.
#
# Emits to $GITHUB_OUTPUT:
#   age_days=<int>
#   should_retrain=<true|false>

LAST=$(git log --format=%aI -1 -- model_assets/level_q50.pkl 2>/dev/null | head -1)
if [ -z "$LAST" ]; then
  AGE=999
else
  NOW=$(date -u +%s)
  THEN=$(date -d "$LAST" +%s 2>/dev/null || echo 0)
  AGE=$(( (NOW - THEN) / 86400 ))
fi
echo "age_days=${AGE}" >> "$GITHUB_OUTPUT"
if [ "$AGE" -ge 7 ]; then
  echo "should_retrain=true"  >> "$GITHUB_OUTPUT"
else
  echo "should_retrain=false" >> "$GITHUB_OUTPUT"
fi
echo "Model age: ${AGE} days — should_retrain=$([ "$AGE" -ge 7 ] && echo true || echo false)"
