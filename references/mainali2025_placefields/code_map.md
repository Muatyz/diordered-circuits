# Code Map: mainali2025_placefields

| Concept | Code | Status |
|---|---|---|
| 1D thresholded GP place fields | `reproduction/src/utils.py::gaussian_process_place_fields_1d` | toy |
| place-map loading and field count | `reproduction/src/compute_place_cell_optimized_weights.py` | implemented |
| optimized recurrent extension | `compute_place_cell_optimized_weights.py::compute_place_cell_weights` | project hypothesis, not paper reproduction |

No paper-specific unit test currently validates the universal field statistics.
