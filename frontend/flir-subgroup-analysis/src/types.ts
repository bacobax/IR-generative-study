export type Phase = "phase1" | "phase2";

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

export interface OptionsResponse {
  dataset: DatasetMetadata;
  constants: {
    analysis_splits: string[];
    size_bin_method: string;
    size_bin_labels: string[];
    fixed_size_bins: number[] | null;
    position_mode: string;
    position_bin_labels: string[];
    dominance_thresholds: number[];
    feasibility_rules: Record<string, number>;
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
