export type Phase = "phase1" | "phase2";
export type DatasetId = "flir_private_proxy_alignment_v18" | "v18";

export interface DatasetOption {
  dataset_id: DatasetId;
  label: string;
  description: string;
  data_root: string;
  is_default: boolean;
}

export interface GroupSpec {
  class_label: string;
  size_bin: string;
  position_bin?: string | null;
}

export interface SelectableGroup extends GroupSpec {
  subgroup_label: string;
  n_instances: number;
  n_images: number;
  median_dominance: number;
}

export interface DatasetMetadata {
  dataset_id: DatasetId;
  label: string;
  description: string;
  data_root: string;
  analysis_splits: string[];
  available_splits: string[];
  n_images: number;
  n_annotations: number;
  n_classes: number;
  classes: string[];
  n_missing_image_files: number;
  layout: Array<Record<string, unknown>>;
  root_metadata: Array<Record<string, unknown>>;
}

export interface OptionsPhasePayload {
  default_group: SelectableGroup;
  example_groups: string[];
  groups: SelectableGroup[];
  dominant_group_overview: Array<Record<string, number>>;
  dominant_group_frequency: Array<Record<string, number | string | null>>;
  feasibility?: Array<Record<string, number | string | boolean | null>>;
  size_bin_spec: Array<Record<string, number | string | null>>;
}

export interface BinExplanationBox {
  ann_id: number;
  class_label: string;
  bbox_x: number;
  bbox_y: number;
  bbox_w: number;
  bbox_h: number;
  bbox_area_norm: number;
  bbox_center_x_norm: number;
  size_bin: string;
  position_bin: string;
}

export interface BinExplanationExample {
  image_key: string;
  image_id: string;
  split: string;
  image_width: number;
  image_height: number;
  covered_bins: string[];
  coverage_count: number;
  n_instances: number;
  selection_reason: string;
  preview_url: string;
  boxes: BinExplanationBox[];
}

export interface BinExplanationPanelPayload {
  panel: "size" | "position";
  bin_labels: string[];
  bin_spec?: Array<Record<string, number | string | null>>;
  bin_edges?: number[];
  example: BinExplanationExample | null;
}

export interface DatasetsResponse {
  default_dataset_id: DatasetId;
  datasets: DatasetOption[];
}

export interface OptionsResponse {
  datasets: DatasetOption[];
  active_dataset: DatasetMetadata;
  constants: {
    analysis_splits: string[];
    size_bin_method: string;
    size_bin_labels: string[];
    fixed_size_bins: number[] | null;
    position_mode: string;
    position_bin_labels: string[];
    position_bin_edges: number[];
    dominance_thresholds: number[];
    feasibility_rules: Record<string, number>;
  };
  bin_explanations: {
    size: BinExplanationPanelPayload;
    position: BinExplanationPanelPayload;
  };
  phase1: OptionsPhasePayload;
  phase2: OptionsPhasePayload;
}

export interface HoldoutCurvePoint {
  subgroup: string;
  tau: number;
  heldout_n_images: number;
  heldout_fraction: number;
  mean_target_count: number;
  median_target_count: number;
  mean_dominance: number;
}

export interface HoldoutCurvesResponse {
  dataset: DatasetId;
  phase: Phase;
  thresholds: number[];
  groups: Array<SelectableGroup & { series: HoldoutCurvePoint[] }>;
}

export interface DominanceHistogramRow {
  bin_start: number;
  bin_end: number;
  count: number;
  bin_label: string;
}

export interface CollateralSummary {
  subgroup: string;
  tau: number;
  heldout_n_images: number;
  collateral_other_loss_frac: number;
}

export interface CollateralGroupPayload extends SelectableGroup {
  summary: CollateralSummary;
  damage_rows: Array<{
    subgroup: string;
    count_before: number;
    count_after: number;
    count_loss: number;
    loss_fraction: number;
    is_target_subgroup: boolean;
  }>;
  dominance_histogram: DominanceHistogramRow[];
}

export interface CollateralResponse {
  dataset: DatasetId;
  phase: Phase;
  tau: number;
  groups: CollateralGroupPayload[];
}

export interface PartitionSummaryRow {
  partition: "train" | "held_out";
  n_images: number;
  mean_total_object_count: number;
  median_total_object_count: number;
  mean_density: number;
  median_density: number;
}

export interface FractionRow {
  partition: "train" | "held_out";
  class_label?: string;
  subgroup?: string;
  count?: number;
  n_images?: number;
  fraction: number;
}

export interface PerClassImageCountRow {
  class_label: string;
  instance_count: number;
  n_images_before: number;
  n_images_after: number;
}

export interface PartitionComparisonsResponse {
  dataset: DatasetId;
  phase: Phase;
  tau: number;
  groups: SelectableGroup[];
  heldout_image_keys: string[];
  heldout_n_images: number;
  numeric_summary: PartitionSummaryRow[];
  class_distribution: FractionRow[];
  class_image_distribution: FractionRow[];
  subgroup_distribution: FractionRow[];
  cooccurring_class_distribution: FractionRow[];
  per_class_image_count_distribution: PerClassImageCountRow[];
}

export interface ExampleBox {
  ann_id: number;
  class_label: string;
  subgroup_label: string;
  bbox_x: number;
  bbox_y: number;
  bbox_w: number;
  bbox_h: number;
  is_target_subgroup: boolean;
  is_target_class: boolean;
}

export interface ExampleImage {
  image_key: string;
  image_id: string;
  partition: string;
  selection_source: string;
  subgroup_label: string;
  dominance_ratio: number;
  subgroup_count: number;
  class_count: number;
  preview_url: string;
  image_width: number;
  image_height: number;
  boxes: ExampleBox[];
}

export interface ExamplesResponse {
  dataset: DatasetId;
  phase: Phase;
  tau: number;
  example_count: number;
  groups: Array<
    SelectableGroup & {
      held_out_examples: ExampleImage[];
      retained_examples: ExampleImage[];
    }
  >;
}

export interface CheckpointSelectionCatalogRequest {
  root: string;
}

export interface CheckpointSelectionRunRequest {
  root: string;
  run: string;
  subroot?: string | null;
}

export interface CheckpointSelectionMetricMap {
  KID?: number | null;
  FID?: number | null;
  MMD?: number | null;
  "Intra-LPIPS"?: number | null;
  [key: string]: number | null | undefined;
}

export interface CheckpointSelectionRunRow {
  subroot: string | null;
  run: string;
  relative_path: string;
  status: string;
  selected_checkpoint: string | null;
  model_type: string | null;
  sampler_name: string | null;
  timestamp: string | null;
  metrics: CheckpointSelectionMetricMap;
  available_preview_stages: string[];
}

export interface CheckpointSelectionCatalogResponse {
  root: string;
  subroots: Array<string | null>;
  runs: CheckpointSelectionRunRow[];
  warnings: string[];
}

export interface CheckpointSelectionPreview {
  checkpoint_identifier: string | null;
  stage: string | null;
  num_preview_images: number | null;
  preview_grid: string | null;
  preview_images: string[];
  tile_size?: number | null;
  columns?: number | null;
  timestamp?: string | null;
}

export interface CheckpointSelectionRunDetail extends CheckpointSelectionRunRow {
  root: string;
  selected_top_checkpoints: string[];
  generation_backend_used: string | null;
  metric_directions: Record<string, string>;
  stage1_ranking: Array<Record<string, number | string | null>>;
  stage2_ranking: Array<Record<string, number | string | null>>;
  final_metrics: Record<string, unknown>;
  summary: Record<string, unknown>;
  cleanup: Record<string, unknown> | null;
  previews: CheckpointSelectionPreview[];
  warnings: string[];
}
