# DWI Neuroplasticity Analysis Scaffold

This repository now includes a Python analysis scaffold that encodes the requested study design for very preterm (VPT) neuroplasticity analyses using fixel-based metrics (FD, FC, FDC), with optional extension to DTI.

## Main module

- `neuroplasticity_analysis.py`

## What is implemented

- Cohort harmonization and required metadata checks.
- SP/TP/BTP performance grouping from DAS-II GCA.
- Explicit FT handling as a separate comparison group.
- FT tract-level reference distribution construction (mean/SD/percentiles).
- FT-referenced z-score derivation for FD/FC/FDC.
- Repair index computation (closeness to FT in vulnerable tracts).
- Compensation index computation (supra-typical enhancement in alternative pathways).
- Group effects and group × Kidokoro moderation model.
- Participant-level repair–compensation continuum table.
- End-to-end convenience runner.

## Notes

- Thresholds are currently configured as:
  - SP: `GCA > 107`
  - TP: `93 <= GCA <= 107`
  - BTP: `GCA < 93`
- Per the coding brief, verify these thresholds against your final Introduction/Methods text before locking analysis.
