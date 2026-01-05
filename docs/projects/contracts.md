# electroflex-contracts

## Purpose
Define shared interfaces and schemas used across ElectroFlex components.

## Exports
- `Forecaster`: common interface for models consumed by eval and API
- `ModelSpec`: stable metadata schema stored with model artifacts

## Why this exists
This package prevents electroflex-eval and electroflex-api from depending on training internals.
Only the contract is shared; implementations remain isolated.