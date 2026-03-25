import {
  startTransition,
  useDeferredValue,
  useEffect,
  useState,
} from "react";
import {
  Bar,
  BarChart,
  CartesianGrid,
  Cell,
  Legend,
  Line,
  LineChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";

import {
  getApiBaseUrl,
  getCollateral,
  getDatasets,
  getExamples,
  getHoldoutCurves,
  getOptions,
  getPartitionComparisons,
} from "./api";
import { InfoHelp } from "./InfoHelp";
import {
  helpTextForAnnotationClassDistribution,
  helpTextForCollateral,
  helpTextForDistinctImagePresence,
  helpTextForDominanceHistogram,
  helpTextForHeldOutExamples,
  helpTextForHoldoutSweep,
  helpTextForPerClassCount,
  helpTextForPositionBinPanel,
  helpTextForRetainedExamples,
  helpTextForSizeBinPanel,
  helpTextForSubgroupDistribution,
  type HelpContext,
} from "./helpText";
import type {
  BinExplanationBox,
  BinExplanationExample,
  BinExplanationPanelPayload,
  CollateralGroupPayload,
  CollateralResponse,
  DatasetId,
  DatasetOption,
  DatasetsResponse,
  ExampleImage,
  ExamplesResponse,
  GroupSpec,
  HoldoutCurvesResponse,
  OptionsPhasePayload,
  OptionsResponse,
  PartitionComparisonsResponse,
  Phase,
  SelectableGroup,
} from "./types";

const EXAMPLE_COUNT = 3;
const SERIES_COLORS = [
  "#c8553d",
  "#3d6cc8",
  "#2f8f68",
  "#ca8a04",
  "#7c3aed",
  "#e11d48",
];

function toGroupSpec(group: SelectableGroup): GroupSpec {
  return {
    class_label: group.class_label,
    size_bin: group.size_bin,
    position_bin: group.position_bin ?? undefined,
  };
}

function formatPercent(value: number): string {
  return `${(value * 100).toFixed(1)}%`;
}

function formatNormalizedRange(value: unknown): string {
  if (typeof value !== "number") {
    return "—";
  }
  if (value === 0) {
    return "0";
  }
  if (value < 0.01) {
    return value.toExponential(2);
  }
  return value.toFixed(3);
}

function buildHoldoutChartRows(holdoutData: HoldoutCurvesResponse | null): Array<Record<string, number | string>> {
  if (!holdoutData) {
    return [];
  }

  const rowsByTau = new Map<number, Record<string, number | string>>();
  holdoutData.groups.forEach((group) => {
    group.series.forEach((point) => {
      const existing = rowsByTau.get(point.tau) ?? { tau: point.tau };
      existing[group.subgroup_label] = point.heldout_n_images;
      rowsByTau.set(point.tau, existing);
    });
  });

  return Array.from(rowsByTau.values()).sort((left, right) => Number(left.tau) - Number(right.tau));
}

function buildPartitionFractionRows(
  rows: PartitionComparisonsResponse["class_distribution"] | PartitionComparisonsResponse["subgroup_distribution"],
  labelKey: "class_label" | "subgroup",
  limit = 12,
): Array<Record<string, number | string>> {
  const totals = new Map<string, number>();
  rows.forEach((row) => {
    const label = row[labelKey];
    if (!label) {
      return;
    }
    const current = totals.get(label) ?? 0;
    totals.set(label, current + (row.count ?? 0));
  });

  const topLabels = Array.from(totals.entries())
    .sort((left, right) => right[1] - left[1])
    .slice(0, limit)
    .map(([label]) => label);

  return topLabels.map((label) => {
    const chartRow: Record<string, number | string> = { label };
    rows
      .filter((row) => row[labelKey] === label)
      .forEach((row) => {
        chartRow[row.partition] = row.fraction;
      });
    chartRow.train = Number(chartRow.train ?? 0);
    chartRow.held_out = Number(chartRow.held_out ?? 0);
    return chartRow;
  });
}

function buildDistinctClassTotals(rows: PartitionComparisonsResponse["per_class_image_count_distribution"]): Array<Record<string, number | string>> {
  const totals = new Map<string, { before: number; after: number }>();
  rows.forEach((row) => {
    const existing = totals.get(row.class_label) ?? { before: 0, after: 0 };
    existing.before += row.n_images_before;
    existing.after += row.n_images_after;
    totals.set(row.class_label, existing);
  });

  return Array.from(totals.entries())
    .map(([classLabel, values]) => ({
      classLabel,
      before: values.before,
      after: values.after,
    }))
    .sort((left, right) => Number(right.before) - Number(left.before));
}

function buildCountDistributionByClass(rows: PartitionComparisonsResponse["per_class_image_count_distribution"]) {
  const grouped = new Map<string, PartitionComparisonsResponse["per_class_image_count_distribution"]>();
  rows.forEach((row) => {
    const existing = grouped.get(row.class_label) ?? [];
    existing.push(row);
    grouped.set(row.class_label, existing);
  });

  return Array.from(grouped.entries())
    .map(([classLabel, classRows]) => ({
      classLabel,
      rows: classRows.sort((left, right) => left.instance_count - right.instance_count),
    }))
    .sort((left, right) => left.classLabel.localeCompare(right.classLabel));
}

function binClassName(label: string): string {
  return label.toLowerCase().replace(/\s+/g, "-");
}

function SectionState({ loading, error }: { loading: boolean; error: string | null }) {
  if (loading) {
    return <div className="status-panel">Loading…</div>;
  }
  if (error) {
    return <div className="status-panel error">{error}</div>;
  }
  return null;
}

function PanelHeader({
  eyebrow,
  title,
  supportingCopy,
  helpText,
  compact = false,
}: {
  eyebrow?: string;
  title: string;
  supportingCopy?: string;
  helpText?: string;
  compact?: boolean;
}) {
  return (
    <div className={`panel-header ${compact ? "panel-header--compact" : ""}`}>
      <div>
        {eyebrow ? <p className="eyebrow">{eyebrow}</p> : null}
        <h3>{title}</h3>
      </div>
      <div className="panel-header__meta">
        {supportingCopy ? <p className="supporting-copy">{supportingCopy}</p> : null}
        {helpText ? <InfoHelp label={`Explain ${title}`} text={helpText} /> : null}
      </div>
    </div>
  );
}

function ExamplePreview({ title, example }: { title: string; example: ExampleImage }) {
  return (
    <article className="example-card">
      <div className="example-card__meta">
        <div>
          <p className="eyebrow">{title}</p>
          <h4>{example.image_key}</h4>
        </div>
        <span className={`source-badge ${example.selection_source}`}>{example.selection_source.split("_").join(" ")}</span>
      </div>
      <div
        className="preview-frame"
        style={{ aspectRatio: `${example.image_width} / ${example.image_height}` }}
      >
        <img
          src={`${getApiBaseUrl()}${example.preview_url}`}
          alt={example.image_key}
          loading="lazy"
        />
        {example.boxes.map((box) => {
          const boxClass = box.is_target_subgroup
            ? "box target-subgroup"
            : box.is_target_class
              ? "box target-class"
              : "box context";
          return (
            <div
              key={`${example.image_key}-${box.ann_id}`}
              className={boxClass}
              title={`${box.class_label} • ${box.subgroup_label}`}
              style={{
                left: `${(box.bbox_x / example.image_width) * 100}%`,
                top: `${(box.bbox_y / example.image_height) * 100}%`,
                width: `${(box.bbox_w / example.image_width) * 100}%`,
                height: `${(box.bbox_h / example.image_height) * 100}%`,
              }}
            />
          );
        })}
      </div>
      <div className="example-card__stats">
        <span>dominance {example.dominance_ratio.toFixed(2)}</span>
        <span>subgroup count {example.subgroup_count}</span>
        <span>class count {example.class_count}</span>
      </div>
    </article>
  );
}

function BinLegend({
  labels,
  mode,
}: {
  labels: string[];
  mode: "size" | "position";
}) {
  return (
    <div className="bin-legend">
      {labels.map((label) => (
        <span key={`${mode}-${label}`} className={`bin-pill ${mode}-${binClassName(label)}`}>
          {label}
        </span>
      ))}
    </div>
  );
}

function BinExplanationOverlay({
  example,
  mode,
  positionEdges,
}: {
  example: BinExplanationExample;
  mode: "size" | "position";
  positionEdges?: number[];
}) {
  return (
    <div
      className="preview-frame preview-frame--bin"
      style={{ aspectRatio: `${example.image_width} / ${example.image_height}` }}
    >
      <img
        src={`${getApiBaseUrl()}${example.preview_url}`}
        alt={example.image_key}
        loading="lazy"
      />
      {mode === "position" && positionEdges ? (
        <div className="position-guides">
          {positionEdges.slice(1, -1).map((edge) => (
            <div
              key={edge}
              className="position-guide"
              style={{ left: `${edge * 100}%` }}
            />
          ))}
        </div>
      ) : null}
      {example.boxes.map((box: BinExplanationBox) => {
        const label = mode === "size" ? box.size_bin : box.position_bin;
        return (
          <div
            key={`${example.image_key}-${box.ann_id}`}
            className={`box box--bin ${mode}-${binClassName(label)}`}
            title={`${box.class_label} • ${label}`}
            style={{
              left: `${(box.bbox_x / example.image_width) * 100}%`,
              top: `${(box.bbox_y / example.image_height) * 100}%`,
              width: `${(box.bbox_w / example.image_width) * 100}%`,
              height: `${(box.bbox_h / example.image_height) * 100}%`,
            }}
          >
            <span className="box__label">{label}</span>
          </div>
        );
      })}
    </div>
  );
}

function BinExplanationPanel({
  title,
  payload,
  helpText,
  datasetLabel,
  positionEdges,
}: {
  title: string;
  payload: BinExplanationPanelPayload;
  helpText: string;
  datasetLabel: string;
  positionEdges?: number[];
}) {
  return (
    <div className="chart-panel">
      <PanelHeader title={title} helpText={helpText} compact />
      {payload.example ? (
        <>
          <BinLegend labels={payload.bin_labels} mode={payload.panel} />
          <BinExplanationOverlay
            example={payload.example}
            mode={payload.panel}
            positionEdges={positionEdges}
          />
          <div className="bin-panel__meta">
            <span>{datasetLabel}</span>
            <span>{payload.example.split} split</span>
            <span>{payload.example.coverage_count} bins covered</span>
            <span>{payload.example.n_instances} instances</span>
          </div>
          <p className="supporting-copy bin-panel__copy">{payload.example.selection_reason}</p>
          {payload.panel === "size" && payload.bin_spec ? (
            <div className="bin-spec-grid">
              {payload.bin_spec.map((row) => (
                <div key={String(row.size_bin)} className="bin-spec-card">
                  <strong>{String(row.size_bin)}</strong>
                  <span>
                    {formatNormalizedRange(row.bin_min)} to {formatNormalizedRange(row.bin_max)}
                  </span>
                  <span>{row.n_instances ?? 0} instances</span>
                </div>
              ))}
            </div>
          ) : null}
        </>
      ) : (
        <div className="status-panel">No representative example image was found for this dataset.</div>
      )}
    </div>
  );
}

function GroupDetailCard({
  group,
  examples,
  damage,
  helpContext,
}: {
  group: SelectableGroup;
  examples: ExamplesResponse["groups"][number] | undefined;
  damage: CollateralGroupPayload | undefined;
  helpContext: HelpContext;
}) {
  const topDamageRows = damage?.damage_rows.slice(0, 12) ?? [];

  return (
    <section className="panel detail-card">
      <div className="detail-card__header">
        <div>
          <p className="eyebrow">Group Detail</p>
          <h3>{group.subgroup_label}</h3>
        </div>
        <div className="detail-card__stats">
          <span>{group.n_images} images</span>
          <span>{group.n_instances} instances</span>
          <span>median dominance {group.median_dominance.toFixed(2)}</span>
        </div>
      </div>

      <div className="detail-chart-grid">
        <div className="chart-panel">
          <PanelHeader
            title="Dominance histogram"
            helpText={helpTextForDominanceHistogram(helpContext, group.subgroup_label)}
            compact
          />
          {damage && damage.dominance_histogram.length > 0 ? (
            <ResponsiveContainer width="100%" height={240}>
              <BarChart data={damage.dominance_histogram}>
                <CartesianGrid strokeDasharray="3 3" vertical={false} />
                <XAxis dataKey="bin_label" interval={2} angle={-25} textAnchor="end" height={72} />
                <YAxis allowDecimals={false} />
                <Tooltip />
                <Bar dataKey="count" fill="#3d6cc8" radius={[8, 8, 0, 0]} />
              </BarChart>
            </ResponsiveContainer>
          ) : (
            <div className="status-panel">No dominance histogram available.</div>
          )}
        </div>

        <div className="chart-panel">
          <PanelHeader
            title="Collateral damage"
            helpText={helpTextForCollateral(helpContext, group.subgroup_label)}
            compact
          />
          {topDamageRows.length > 0 ? (
            <ResponsiveContainer width="100%" height={240}>
              <BarChart data={topDamageRows} layout="vertical" margin={{ left: 20 }}>
                <CartesianGrid strokeDasharray="3 3" horizontal={false} />
                <XAxis type="number" tickFormatter={(value) => `${Math.round(value * 100)}%`} />
                <YAxis
                  type="category"
                  dataKey="subgroup"
                  width={190}
                  tickFormatter={(value) => String(value).replace("class=", "")}
                />
                <Tooltip formatter={(value: number) => `${(value * 100).toFixed(1)}%`} />
                <Bar dataKey="loss_fraction" radius={[0, 8, 8, 0]}>
                  {topDamageRows.map((row) => (
                    <Cell
                      key={`${group.subgroup_label}-${row.subgroup}`}
                      fill={row.is_target_subgroup ? "#c8553d" : "#2f8f68"}
                    />
                  ))}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          ) : (
            <div className="status-panel">No collateral rows available.</div>
          )}
        </div>
      </div>

      <div className="examples-grid">
        <div className="examples-column">
          <div className="examples-column__header">
            <div className="examples-column__title">
              <h4>Held-out examples</h4>
              <InfoHelp
                label={`Explain held-out examples for ${group.subgroup_label}`}
                text={helpTextForHeldOutExamples(helpContext, group.subgroup_label)}
              />
            </div>
            <span>{examples?.held_out_examples.length ?? 0} images</span>
          </div>
          {examples && examples.held_out_examples.length > 0 ? (
            examples.held_out_examples.map((example, index) => (
              <ExamplePreview key={`${example.image_key}-${index}`} title={`held out #${index + 1}`} example={example} />
            ))
          ) : (
            <div className="status-panel">No held-out examples at this tau.</div>
          )}
        </div>

        <div className="examples-column">
          <div className="examples-column__header">
            <div className="examples-column__title">
              <h4>Retained training examples</h4>
              <InfoHelp
                label={`Explain retained examples for ${group.subgroup_label}`}
                text={helpTextForRetainedExamples(helpContext, group.subgroup_label)}
              />
            </div>
            <span>{examples?.retained_examples.length ?? 0} images</span>
          </div>
          {examples && examples.retained_examples.length > 0 ? (
            examples.retained_examples.map((example, index) => (
              <ExamplePreview key={`${example.image_key}-${index}`} title={`retained #${index + 1}`} example={example} />
            ))
          ) : (
            <div className="status-panel">No retained examples for this group.</div>
          )}
        </div>
      </div>
    </section>
  );
}

export default function App() {
  const [datasetCatalog, setDatasetCatalog] = useState<DatasetOption[]>([]);
  const [selectedDataset, setSelectedDataset] = useState<DatasetId | null>(null);
  const [datasetsLoading, setDatasetsLoading] = useState(true);
  const [datasetsError, setDatasetsError] = useState<string | null>(null);

  const [options, setOptions] = useState<OptionsResponse | null>(null);
  const [optionsLoading, setOptionsLoading] = useState(false);
  const [optionsError, setOptionsError] = useState<string | null>(null);

  const [phase, setPhase] = useState<Phase>("phase1");
  const [selectedGroupLabels, setSelectedGroupLabels] = useState<string[]>([]);
  const [tau, setTau] = useState(0.5);

  const [classFilter, setClassFilter] = useState("all");
  const [sizeFilter, setSizeFilter] = useState("all");
  const [positionFilter, setPositionFilter] = useState("all");

  const [holdoutData, setHoldoutData] = useState<HoldoutCurvesResponse | null>(null);
  const [holdoutLoading, setHoldoutLoading] = useState(false);
  const [holdoutError, setHoldoutError] = useState<string | null>(null);

  const [collateralData, setCollateralData] = useState<CollateralResponse | null>(null);
  const [collateralLoading, setCollateralLoading] = useState(false);
  const [collateralError, setCollateralError] = useState<string | null>(null);

  const [partitionData, setPartitionData] = useState<PartitionComparisonsResponse | null>(null);
  const [partitionLoading, setPartitionLoading] = useState(false);
  const [partitionError, setPartitionError] = useState<string | null>(null);

  const [examplesData, setExamplesData] = useState<ExamplesResponse | null>(null);
  const [examplesLoading, setExamplesLoading] = useState(false);
  const [examplesError, setExamplesError] = useState<string | null>(null);

  useEffect(() => {
    let cancelled = false;

    async function loadDatasets() {
      setDatasetsLoading(true);
      setDatasetsError(null);
      try {
        const payload: DatasetsResponse = await getDatasets();
        if (cancelled) {
          return;
        }
        setDatasetCatalog(payload.datasets);
        setSelectedDataset((current) => current ?? payload.default_dataset_id);
      } catch (error) {
        if (!cancelled) {
          setDatasetsError(error instanceof Error ? error.message : "Failed to load datasets");
        }
      } finally {
        if (!cancelled) {
          setDatasetsLoading(false);
        }
      }
    }

    void loadDatasets();
    return () => {
      cancelled = true;
    };
  }, []);

  useEffect(() => {
    if (!selectedDataset) {
      return;
    }

    const datasetId = selectedDataset;
    let cancelled = false;

    async function loadOptions() {
      setOptionsLoading(true);
      setOptionsError(null);
      try {
        const payload = await getOptions(datasetId);
        if (!cancelled) {
          setOptions(payload);
          setDatasetCatalog(payload.datasets);
        }
      } catch (error) {
        if (!cancelled) {
          setOptionsError(error instanceof Error ? error.message : "Failed to load options");
        }
      } finally {
        if (!cancelled) {
          setOptionsLoading(false);
        }
      }
    }

    void loadOptions();
    return () => {
      cancelled = true;
    };
  }, [selectedDataset]);

  const phaseOptions: OptionsPhasePayload | null =
    options == null ? null : phase === "phase1" ? options.phase1 : options.phase2;

  useEffect(() => {
    if (!phaseOptions) {
      return;
    }

    const validLabels = new Set(phaseOptions.groups.map((group) => group.subgroup_label));
    const stillValid = selectedGroupLabels.filter((label) => validLabels.has(label));
    if (stillValid.length > 0) {
      if (stillValid.length !== selectedGroupLabels.length) {
        setSelectedGroupLabels(stillValid);
      }
      return;
    }

    setSelectedGroupLabels([phaseOptions.default_group.subgroup_label]);
    if (phase === "phase1") {
      setPositionFilter("all");
    }
  }, [phase, phaseOptions, selectedGroupLabels]);

  useEffect(() => {
    const availableClasses = options?.active_dataset.classes ?? [];
    if (availableClasses.length === 1) {
      setClassFilter(availableClasses[0]);
      return;
    }
    if (classFilter !== "all" && !availableClasses.includes(classFilter)) {
      setClassFilter("all");
    }
  }, [classFilter, options]);

  const deferredClassFilter = useDeferredValue(classFilter);
  const deferredSizeFilter = useDeferredValue(sizeFilter);
  const deferredPositionFilter = useDeferredValue(positionFilter);

  const filteredGroups = phaseOptions?.groups.filter((group) => {
    const classMatch = deferredClassFilter === "all" || group.class_label === deferredClassFilter;
    const sizeMatch = deferredSizeFilter === "all" || group.size_bin === deferredSizeFilter;
    const positionMatch =
      phase === "phase1" || deferredPositionFilter === "all" || group.position_bin === deferredPositionFilter;
    return classMatch && sizeMatch && positionMatch;
  }) ?? [];

  const selectedGroups =
    phaseOptions?.groups.filter((group) => selectedGroupLabels.includes(group.subgroup_label)) ?? [];
  const selectionKey = `${selectedDataset ?? "none"}::${selectedGroupLabels.join("||")}`;

  useEffect(() => {
    if (!selectedDataset) {
      return;
    }

    const requestGroups = phaseOptions?.groups.filter((group) => selectedGroupLabels.includes(group.subgroup_label)) ?? [];

    if (!phaseOptions || requestGroups.length === 0) {
      setHoldoutData(null);
      setCollateralData(null);
      setPartitionData(null);
      setExamplesData(null);
      return;
    }

    let cancelled = false;
    const groupSpecs = requestGroups.map(toGroupSpec);

    setHoldoutLoading(true);
    setHoldoutError(null);
    void getHoldoutCurves({ dataset: selectedDataset, phase, groups: groupSpecs })
      .then((payload) => {
        if (!cancelled) {
          setHoldoutData(payload);
        }
      })
      .catch((error) => {
        if (!cancelled) {
          setHoldoutError(error instanceof Error ? error.message : "Failed to load holdout curves");
        }
      })
      .finally(() => {
        if (!cancelled) {
          setHoldoutLoading(false);
        }
      });

    setCollateralLoading(true);
    setCollateralError(null);
    void getCollateral({ dataset: selectedDataset, phase, groups: groupSpecs, tau })
      .then((payload) => {
        if (!cancelled) {
          setCollateralData(payload);
        }
      })
      .catch((error) => {
        if (!cancelled) {
          setCollateralError(error instanceof Error ? error.message : "Failed to load collateral analysis");
        }
      })
      .finally(() => {
        if (!cancelled) {
          setCollateralLoading(false);
        }
      });

    setPartitionLoading(true);
    setPartitionError(null);
    void getPartitionComparisons({ dataset: selectedDataset, phase, groups: groupSpecs, tau, include_zero_counts: false })
      .then((payload) => {
        if (!cancelled) {
          setPartitionData(payload);
        }
      })
      .catch((error) => {
        if (!cancelled) {
          setPartitionError(error instanceof Error ? error.message : "Failed to load partition comparisons");
        }
      })
      .finally(() => {
        if (!cancelled) {
          setPartitionLoading(false);
        }
      });

    setExamplesLoading(true);
    setExamplesError(null);
    void getExamples({ dataset: selectedDataset, phase, groups: groupSpecs, tau, example_count: EXAMPLE_COUNT })
      .then((payload) => {
        if (!cancelled) {
          setExamplesData(payload);
        }
      })
      .catch((error) => {
        if (!cancelled) {
          setExamplesError(error instanceof Error ? error.message : "Failed to load examples");
        }
      })
      .finally(() => {
        if (!cancelled) {
          setExamplesLoading(false);
        }
      });

    return () => {
      cancelled = true;
    };
  }, [phase, phaseOptions, selectedDataset, selectionKey, selectedGroupLabels, tau]);

  const holdoutChartRows = buildHoldoutChartRows(holdoutData);
  const classDistributionRows = partitionData
    ? buildPartitionFractionRows(partitionData.class_distribution, "class_label")
    : [];
  const subgroupDistributionRows = partitionData
    ? buildPartitionFractionRows(partitionData.subgroup_distribution, "subgroup")
    : [];
  const distinctClassTotals = partitionData
    ? buildDistinctClassTotals(partitionData.per_class_image_count_distribution)
    : [];
  const perClassCountCharts = partitionData
    ? buildCountDistributionByClass(partitionData.per_class_image_count_distribution)
    : [];

  const activeDataset = options?.active_dataset ?? null;
  const availableClasses = activeDataset?.classes ?? [];
  const availableSizes = options?.constants.size_bin_labels ?? [];
  const availablePositions = options?.constants.position_bin_labels ?? [];
  const datasetLabel = activeDataset?.label ?? "Selected dataset";
  const datasetDescription = activeDataset?.description ?? "";
  const helpContext: HelpContext = {
    datasetLabel,
    phase,
    tau,
    selectedGroupLabels,
    classCount: activeDataset?.n_classes ?? 0,
    heldoutCount: partitionData?.heldout_n_images,
  };

  function handlePhaseChange(nextPhase: Phase) {
    startTransition(() => {
      setPhase(nextPhase);
      setSelectedGroupLabels([]);
    });
  }

  function handleDatasetChange(nextDataset: DatasetId) {
    startTransition(() => {
      setSelectedDataset(nextDataset);
      setSelectedGroupLabels([]);
      setClassFilter("all");
      setSizeFilter("all");
      setPositionFilter("all");
    });
  }

  function toggleGroupSelection(groupLabel: string) {
    startTransition(() => {
      setSelectedGroupLabels((current) =>
        current.includes(groupLabel)
          ? current.filter((label) => label !== groupLabel)
          : [...current, groupLabel],
      );
    });
  }

  function selectFilteredGroups() {
    startTransition(() => {
      setSelectedGroupLabels(filteredGroups.map((group) => group.subgroup_label));
    });
  }

  function clearSelectedGroups() {
    startTransition(() => {
      setSelectedGroupLabels([]);
    });
  }

  const datasetsReady = !datasetsLoading && !datasetsError;
  const classSelectorLocked = availableClasses.length === 1;

  return (
    <div className="app-shell">
      <header className="hero">
        <div className="hero__copy">
          <p className="eyebrow">Subgroup Analysis Workspace</p>
          <h1>Notebook-grounded hold-out analysis across FLIR-style datasets</h1>
          <p className="hero__lede">
            Switch datasets, compare subgroup hold-out rules, inspect collateral damage, and review held-out versus
            retained examples without leaving the live analysis app.
          </p>
        </div>
        <div className="hero__stats">
          <div className="stat-card">
            <span>dataset</span>
            <strong>{activeDataset?.label ?? "—"}</strong>
          </div>
          <div className="stat-card">
            <span>images</span>
            <strong>{activeDataset?.n_images ?? "—"}</strong>
          </div>
          <div className="stat-card">
            <span>classes</span>
            <strong>{activeDataset?.n_classes ?? "—"}</strong>
          </div>
        </div>
      </header>

      <SectionState loading={datasetsLoading} error={datasetsError} />
      <SectionState loading={optionsLoading} error={optionsError} />

      {datasetsReady && options && phaseOptions && selectedDataset ? (
        <>
          <section className="panel controls-panel">
            <div className="controls-header">
              <div>
                <p className="eyebrow">Controls</p>
                <h2>Choose the dataset, subgroup regime, and tau</h2>
              </div>
              <div className="phase-toggle">
                <button
                  type="button"
                  className={phase === "phase1" ? "active" : ""}
                  onClick={() => handlePhaseChange("phase1")}
                >
                  Phase 1
                </button>
                <button
                  type="button"
                  className={phase === "phase2" ? "active" : ""}
                  onClick={() => handlePhaseChange("phase2")}
                >
                  Phase 2
                </button>
              </div>
            </div>

            <div className="controls-grid">
              <label>
                <span>Dataset</span>
                <select
                  value={selectedDataset}
                  onChange={(event) => handleDatasetChange(event.target.value as DatasetId)}
                >
                  {datasetCatalog.map((dataset) => (
                    <option key={dataset.dataset_id} value={dataset.dataset_id}>
                      {dataset.label}
                    </option>
                  ))}
                </select>
              </label>

              {classSelectorLocked ? (
                <label className="disabled-control">
                  <span>Class</span>
                  <input value={availableClasses[0] ?? "No class"} disabled />
                </label>
              ) : (
                <label>
                  <span>Class</span>
                  <select value={classFilter} onChange={(event) => setClassFilter(event.target.value)}>
                    <option value="all">All classes</option>
                    {availableClasses.map((classLabel) => (
                      <option key={classLabel} value={classLabel}>
                        {classLabel}
                      </option>
                    ))}
                  </select>
                </label>
              )}

              <label>
                <span>Size bin</span>
                <select value={sizeFilter} onChange={(event) => setSizeFilter(event.target.value)}>
                  <option value="all">All size bins</option>
                  {availableSizes.map((sizeLabel) => (
                    <option key={sizeLabel} value={sizeLabel}>
                      {sizeLabel}
                    </option>
                  ))}
                </select>
              </label>

              {phase === "phase2" ? (
                <label>
                  <span>Position bin</span>
                  <select value={positionFilter} onChange={(event) => setPositionFilter(event.target.value)}>
                    <option value="all">All positions</option>
                    {availablePositions.map((positionLabel) => (
                      <option key={positionLabel} value={positionLabel}>
                        {positionLabel}
                      </option>
                    ))}
                  </select>
                </label>
              ) : (
                <label className="disabled-control">
                  <span>Position bin</span>
                  <input value="Phase 2 only" disabled />
                </label>
              )}

              <label className="tau-control controls-grid__span-2">
                <span>Tau threshold</span>
                <div className="tau-control__inputs">
                  <input
                    type="range"
                    min={0.1}
                    max={0.9}
                    step={0.05}
                    value={tau}
                    onChange={(event) => setTau(Number(event.target.value))}
                  />
                  <input
                    type="number"
                    min={0}
                    max={1}
                    step={0.05}
                    value={tau}
                    onChange={(event) => setTau(Number(event.target.value))}
                  />
                </div>
              </label>
            </div>

            <p className="supporting-copy controls-panel__dataset-copy">
              {datasetDescription}
            </p>

            <div className="selection-toolbar">
              <div className="selection-toolbar__actions">
                <button type="button" onClick={selectFilteredGroups}>
                  Select filtered groups
                </button>
                <button type="button" className="secondary" onClick={clearSelectedGroups}>
                  Clear selection
                </button>
              </div>
              <div className="selection-toolbar__chips">
                {selectedGroups.map((group) => (
                  <button
                    key={group.subgroup_label}
                    type="button"
                    className="chip"
                    onClick={() => toggleGroupSelection(group.subgroup_label)}
                  >
                    {group.subgroup_label}
                  </button>
                ))}
              </div>
            </div>

            <div className="group-table">
              <table>
                <thead>
                  <tr>
                    <th>Use</th>
                    <th>Class</th>
                    <th>Size</th>
                    {phase === "phase2" ? <th>Position</th> : null}
                    <th>Canonical label</th>
                    <th>Images</th>
                    <th>Instances</th>
                    <th>Median dominance</th>
                  </tr>
                </thead>
                <tbody>
                  {filteredGroups.map((group) => (
                    <tr key={group.subgroup_label}>
                      <td>
                        <input
                          type="checkbox"
                          checked={selectedGroupLabels.includes(group.subgroup_label)}
                          onChange={() => toggleGroupSelection(group.subgroup_label)}
                        />
                      </td>
                      <td>{group.class_label}</td>
                      <td>{group.size_bin}</td>
                      {phase === "phase2" ? <td>{group.position_bin}</td> : null}
                      <td>{group.subgroup_label}</td>
                      <td>{group.n_images}</td>
                      <td>{group.n_instances}</td>
                      <td>{group.median_dominance.toFixed(2)}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </section>

          <section className="panel">
            <PanelHeader
              eyebrow="Bin Guide"
              title="How the backend assigns size bins and position bins"
              supportingCopy="The visual examples below come from the selected dataset and use the same binning rules that power phase 1 and phase 2 subgroup construction."
            />
            <div className="detail-chart-grid">
              <BinExplanationPanel
                title="Size bins"
                payload={options.bin_explanations.size}
                helpText={helpTextForSizeBinPanel(datasetLabel)}
                datasetLabel={datasetLabel}
              />
              <BinExplanationPanel
                title="Position bins"
                payload={options.bin_explanations.position}
                helpText={helpTextForPositionBinPanel(datasetLabel)}
                datasetLabel={datasetLabel}
                positionEdges={options.constants.position_bin_edges}
              />
            </div>
          </section>

          <section className="panel">
            <PanelHeader
              eyebrow="Held-out size sweep"
              title="Held-out images vs tau"
              supportingCopy="Every line uses the notebook hold-out rule m_g(x) >= 1 and r_g(x) >= tau."
              helpText={helpTextForHoldoutSweep(helpContext)}
            />
            <SectionState loading={holdoutLoading} error={holdoutError} />
            {holdoutChartRows.length > 0 ? (
              <div className="chart-panel chart-panel--tall">
                <ResponsiveContainer width="100%" height={360}>
                  <LineChart data={holdoutChartRows}>
                    <CartesianGrid strokeDasharray="3 3" />
                    <XAxis dataKey="tau" />
                    <YAxis allowDecimals={false} />
                    <Tooltip />
                    <Legend />
                    {holdoutData?.groups.map((group, index) => (
                      <Line
                        key={group.subgroup_label}
                        type="monotone"
                        dataKey={group.subgroup_label}
                        stroke={SERIES_COLORS[index % SERIES_COLORS.length]}
                        strokeWidth={3}
                        dot={{ r: 3 }}
                      />
                    ))}
                  </LineChart>
                </ResponsiveContainer>
              </div>
            ) : (
              <div className="status-panel">Select one or more groups to see the tau sweep.</div>
            )}
          </section>

          <section className="panel">
            <PanelHeader
              eyebrow="Union partition view"
              title="Before vs after hold-out across the selected group set"
              supportingCopy="This section applies the union of held-out images across every selected subgroup and compares the train and held-out partitions."
            />
            <SectionState loading={partitionLoading} error={partitionError} />

            {partitionData ? (
              <>
                <div className="summary-grid">
                  {partitionData.numeric_summary.map((row) => (
                    <div key={row.partition} className="summary-card">
                      <h3>{row.partition === "held_out" ? "Held-out" : "Train"}</h3>
                      <p>{row.n_images} distinct images</p>
                      <p>mean objects {row.mean_total_object_count.toFixed(2)}</p>
                      <p>median density {row.median_density.toFixed(3)}</p>
                    </div>
                  ))}
                  <div className="summary-card accent">
                    <h3>Union hold-out</h3>
                    <p>{partitionData.heldout_n_images} images removed</p>
                    <p>{selectedGroups.length} selected groups</p>
                    <p>tau {tau.toFixed(2)}</p>
                  </div>
                </div>

                <div className="detail-chart-grid">
                  <div className="chart-panel">
                    <PanelHeader
                      title="Distinct image presence by class"
                      helpText={helpTextForDistinctImagePresence(helpContext)}
                      compact
                    />
                    <ResponsiveContainer width="100%" height={320}>
                      <BarChart data={distinctClassTotals}>
                        <CartesianGrid strokeDasharray="3 3" vertical={false} />
                        <XAxis dataKey="classLabel" angle={-25} textAnchor="end" height={72} />
                        <YAxis allowDecimals={false} />
                        <Tooltip />
                        <Legend />
                        <Bar dataKey="before" name="Before hold-out" fill="#3d6cc8" radius={[8, 8, 0, 0]} />
                        <Bar dataKey="after" name="After hold-out" fill="#c8553d" radius={[8, 8, 0, 0]} />
                      </BarChart>
                    </ResponsiveContainer>
                  </div>

                  <div className="chart-panel">
                    <PanelHeader
                      title="Notebook annotation-class distribution"
                      helpText={helpTextForAnnotationClassDistribution(helpContext)}
                      compact
                    />
                    <ResponsiveContainer width="100%" height={320}>
                      <BarChart data={classDistributionRows}>
                        <CartesianGrid strokeDasharray="3 3" vertical={false} />
                        <XAxis dataKey="label" angle={-25} textAnchor="end" height={72} />
                        <YAxis tickFormatter={(value) => `${Math.round(value * 100)}%`} />
                        <Tooltip formatter={(value: number) => formatPercent(value)} />
                        <Legend />
                        <Bar dataKey="train" name="Train fraction" fill="#2f8f68" radius={[8, 8, 0, 0]} />
                        <Bar dataKey="held_out" name="Held-out fraction" fill="#ca8a04" radius={[8, 8, 0, 0]} />
                      </BarChart>
                    </ResponsiveContainer>
                  </div>

                  <div className="chart-panel">
                    <PanelHeader
                      title="Notebook subgroup distribution"
                      helpText={helpTextForSubgroupDistribution(helpContext)}
                      compact
                    />
                    <ResponsiveContainer width="100%" height={320}>
                      <BarChart data={subgroupDistributionRows}>
                        <CartesianGrid strokeDasharray="3 3" vertical={false} />
                        <XAxis dataKey="label" angle={-25} textAnchor="end" height={88} />
                        <YAxis tickFormatter={(value) => `${Math.round(value * 100)}%`} />
                        <Tooltip formatter={(value: number) => formatPercent(value)} />
                        <Legend />
                        <Bar dataKey="train" name="Train fraction" fill="#3d6cc8" radius={[8, 8, 0, 0]} />
                        <Bar dataKey="held_out" name="Held-out fraction" fill="#c8553d" radius={[8, 8, 0, 0]} />
                      </BarChart>
                    </ResponsiveContainer>
                  </div>
                </div>

                <div className="count-grid">
                  {perClassCountCharts.map(({ classLabel, rows }) => (
                    <div key={classLabel} className="count-card">
                      <PanelHeader
                        title={classLabel}
                        helpText={helpTextForPerClassCount(helpContext, classLabel)}
                        compact
                      />
                      <ResponsiveContainer width="100%" height={220}>
                        <BarChart data={rows}>
                          <CartesianGrid strokeDasharray="3 3" vertical={false} />
                          <XAxis dataKey="instance_count" allowDecimals={false} />
                          <YAxis allowDecimals={false} />
                          <Tooltip />
                          <Legend />
                          <Bar dataKey="n_images_before" name="Before" fill="#3d6cc8" radius={[8, 8, 0, 0]} />
                          <Bar dataKey="n_images_after" name="After" fill="#c8553d" radius={[8, 8, 0, 0]} />
                        </BarChart>
                      </ResponsiveContainer>
                    </div>
                  ))}
                </div>
              </>
            ) : null}
          </section>

          <section className="details-stack">
            <SectionState loading={collateralLoading || examplesLoading} error={collateralError ?? examplesError} />
            {selectedGroups.map((group) => (
              <GroupDetailCard
                key={group.subgroup_label}
                group={group}
                examples={examplesData?.groups.find((entry) => entry.subgroup_label === group.subgroup_label)}
                damage={collateralData?.groups.find((entry) => entry.subgroup_label === group.subgroup_label)}
                helpContext={helpContext}
              />
            ))}
          </section>
        </>
      ) : null}
    </div>
  );
}
