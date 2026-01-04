# ElectroFlex — System Architecture (Skeleton)

ElectroFlex is an applied AI system for forecasting electricity demand
and enabling flexible consumption through optimisation and control.

This repository is organised as a monorepo with four core projects:

- electroflex-data   : data ingestion and preprocessing
- electroflex-train  : model training and artifact creation
- electroflex-eval   : offline evaluation and quality checks
- electroflex-api    : inference API exposing predictions

Each project defines clear input/output contracts and is tested independently.
Integration tests ensure the system remains connected end-to-end.
