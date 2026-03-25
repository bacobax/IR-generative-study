import type { Phase } from "./types";

export interface HelpContext {
  datasetLabel: string;
  phase: Phase;
  tau: number;
  selectedGroupLabels: string[];
  classCount: number;
  heldoutCount?: number;
}

function groupPhrase(labels: string[]): string {
  if (labels.length === 0) {
    return "no held-out groups are currently selected";
  }
  if (labels.length === 1) {
    return `the selected held-out group ${labels[0]}`;
  }
  return `${labels.length} selected held-out groups`;
}

function phaseLabel(phase: Phase): string {
  return phase === "phase1" ? "phase 1 tuples" : "phase 2 triplets";
}

export function helpTextForHoldoutSweep(context: HelpContext): string {
  return `This line chart uses the selected dataset ${context.datasetLabel} and ${phaseLabel(context.phase)} to sweep the notebook hold-out rule across tau thresholds for ${groupPhrase(context.selectedGroupLabels)}. Each line shows how many distinct train-split images would move into the held-out split for one subgroup when that image contains the subgroup at least once and its dominance ratio meets the threshold on the x-axis.`;
}

export function helpTextForDistinctImagePresence(context: HelpContext): string {
  return `This grouped bar chart counts distinct train-split images in ${context.datasetLabel} before hold-out and after removing the union of images held out by ${groupPhrase(context.selectedGroupLabels)} at tau ${context.tau.toFixed(2)}. It shows how many images still contain each class after the selected hold-out rule is applied, not how many annotations belong to that class.`;
}

export function helpTextForAnnotationClassDistribution(context: HelpContext): string {
  return `This chart compares annotation-class fractions between the remaining train split and the held-out split produced from ${groupPhrase(context.selectedGroupLabels)} at tau ${context.tau.toFixed(2)} in ${context.datasetLabel}. Each bar is the fraction of annotations assigned to one class inside its partition, so the held-out and train bars reveal which classes become overrepresented after the split.`;
}

export function helpTextForSubgroupDistribution(context: HelpContext): string {
  return `This chart compares subgroup annotation fractions between the remaining train split and the held-out split for ${groupPhrase(context.selectedGroupLabels)} at tau ${context.tau.toFixed(2)} in ${context.datasetLabel}. Each subgroup bar uses the canonical notebook label, so you can see which class-size or class-size-position combinations gain or lose share when the selected hold-out rule is applied.`;
}

export function helpTextForPerClassCount(context: HelpContext, classLabel: string): string {
  return `This chart is restricted to the class ${classLabel} in ${context.datasetLabel}. For each instance count on the x-axis, it shows how many distinct train-split images contained exactly that many ${classLabel} instances before hold-out and how many remain in the train split after removing the union hold-out defined by ${groupPhrase(context.selectedGroupLabels)} at tau ${context.tau.toFixed(2)}.`;
}

export function helpTextForDominanceHistogram(context: HelpContext, subgroupLabel: string): string {
  return `This histogram uses ${context.datasetLabel} and the selected ${phaseLabel(context.phase)} subgroup ${subgroupLabel}. Each bar counts train-split images that contain the subgroup and fall into one dominance-ratio interval, where dominance ratio is subgroup count divided by total object count for that image.`;
}

export function helpTextForCollateral(context: HelpContext, subgroupLabel: string): string {
  return `This bar chart measures collateral damage for the subgroup ${subgroupLabel} in ${context.datasetLabel} at tau ${context.tau.toFixed(2)}. It compares subgroup annotation counts before hold-out versus after removing images held out by that subgroup alone, so each bar shows the fraction of annotations lost by another subgroup when the target hold-out rule is enforced.`;
}

export function helpTextForHeldOutExamples(context: HelpContext, subgroupLabel: string): string {
  return `These images come from the selected dataset ${context.datasetLabel} and are held out because they contain the subgroup ${subgroupLabel} in the train split with subgroup count at least 1 and dominance ratio at least tau ${context.tau.toFixed(2)}. The examples are chosen deterministically across the subgroup dominance distribution so you can see low, middle, and high dominance cases for the same held-out rule.`;
}

export function helpTextForRetainedExamples(context: HelpContext, subgroupLabel: string): string {
  return `These images remain in the train split of ${context.datasetLabel} after applying the hold-out rule for ${subgroupLabel} at tau ${context.tau.toFixed(2)}. The backend first selects retained images that still contain the exact subgroup, and if too few remain it backfills with retained images from the same class so the examples stay class-matched while the selection_source badge makes the fallback explicit.`;
}

export function helpTextForSizeBinPanel(datasetLabel: string): string {
  return `This panel uses one representative train-split image from ${datasetLabel} and overlays real annotated boxes colored by the size bin assigned by the backend. The size labels small, medium, and large come from the same normalized bounding-box area quantiles used for subgroup construction, and the bin range table below shows the exact area intervals computed for the selected dataset.`;
}

export function helpTextForPositionBinPanel(datasetLabel: string): string {
  return `This panel uses one representative train-split image from ${datasetLabel} and overlays the exact horizontal thirds used by the backend to assign left, center, and right position bins. Each box is labeled with the same position bin that phase 2 uses when forming class-size-position triplets, so the visual guide matches the implemented subgroup logic exactly.`;
}
