# Ziwei UI Notes

## Component Structure

- `app/ui/ziwei_panel.py`
- Main panel class: `ZiweiPanel`
- UI blocks:
  - Input area (birth info)
  - View controls (triad/four-transform, natal/decadal/yearly, filters)
  - Summary bar
  - Chart area (4x4 ring layout)
  - Detail panel
  - Status line

## Data/Selector Separation

- Raw payload creation:
  - `app/ziwei_service.py::compute_ziwei_payload`
- UI derived model:
  - `app/ziwei_service.py::build_chart_view_model`
- The panel stores raw payload and only re-renders derived view model on UI state changes.

## Main UI State

- `view_mode`: `triad | four_transform`
- `scope`: `natal | decadal | yearly`
- `star_filter`: `major | major_minor | all`
- `show_hanja`: `bool`
- `selected_palace_index`: currently selected palace

## Responsive Strategy

- Desktop (`>= 1180px`): chart (left) + detail panel (right)
- Compact (`< 1180px`): chart on top + detail panel below
- Controlled in `ZiweiPanel._on_resize`

## Extensibility Points

- Replace payload generator with real engine adapter:
  - Keep `compute_ziwei_payload` output contract.
- Add richer relation visualization:
  - Extend `build_chart_view_model` with graph edges.
- Add persistence/history:
  - Save raw payload + UI state into `history_store` extension.
- Add interpretation layer:
  - Keep in separate module consuming `selected_detail`.
