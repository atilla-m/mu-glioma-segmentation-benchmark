# Notebook selection

This folder contains only clean execution notebooks used for the accepted
experimental matrix. Unused platform variants and output-bearing executed
copies are excluded.

`execution_notebook_manifest.csv` records the original project path, execution
platform, SHA-256 checksum, frozen canonical notebook checksum, and current run
status for every file.

There are 36 notebook files for 35 experiment IDs because `no30` genuinely
began on a T4 and resumed on an A40. Both stages are necessary to document the
resumed run; they do not represent two independent experiments.
