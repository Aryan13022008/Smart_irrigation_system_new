# Phase 20 — Schedule-Conditioned Forecasting Feasibility Audit

## Purpose & Scope
This directory contains the machine-readable artifacts and documentation for **Phase 20: Schedule-Conditioned Forecasting Feasibility Audit**.

The objective of this phase is to evaluate whether adding planned irrigation schedule information could improve the experimental 24-hour soil-moisture forecasting system, what schedule data actually exists in the current project, and how a future schedule-conditioned model could be evaluated scientifically.

## Non-Destructive Audit Declarations
- **No Model Training:** Zero models were trained, retrained, or fine-tuned.
- **No Production Modifications:** The production classifier (`training/models/random_forest_11feat.pkl`, threshold `0.3271`), feature order, database, firmware, and dashboard remain 100% untouched.
- **Safety Hard-Lock:** `PUMP_ALLOWED = False` is strictly preserved and enforced across all modules. No GPIO relay actuation, pump commands, or automated valve controls were added.
- **Artifact Isolation:** All Phase 20 artifacts are strictly confined to `training/experimental/phase20/` and `docs/`.

## Artifact Inventory
1. `training/experimental/phase20/phase20_feasibility.json`
   Machine-readable inventory of schedule sources, feature availability matrix, Phase 19 causal selector verification, proposed model architectures, evaluation protocols, and decision criteria.
2. `training/experimental/phase20/README.md`
   This file, providing operational scope and verification declarations.
3. `docs/phase20_schedule_conditioned_feasibility.md`
   Comprehensive analytical report detailing the schedule inventory, forecast information boundaries, Phase 19 causal audit, candidate experimental models, evaluation protocols, data collection schema, and go/no-go recommendation.
