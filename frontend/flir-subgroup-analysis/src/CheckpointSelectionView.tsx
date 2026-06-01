import { useEffect, useMemo, useState } from "react";
import {
  Bar,
  BarChart,
  CartesianGrid,
  Legend,
  Line,
  LineChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";

import {
  getCheckpointSelectionCatalog,
  getCheckpointSelectionPreviewUrl,
  getCheckpointSelectionRun,
} from "./api";
import type {
  CheckpointSelectionCatalogResponse,
  CheckpointSelectionMetricMap,
  CheckpointSelectionPreview,
  CheckpointSelectionRunDetail,
  CheckpointSelectionRunRow,
} from "./types";

const DEFAULT_ROOT = "artifacts/generated/checkpoint_selection";
const METRIC_KEYS = ["KID", "FID", "MMD", "Intra-LPIPS"] as const;
const ROOT_FILTER = "__root__";
const ALL_FILTER = "__all__";

type MetricKey = (typeof METRIC_KEYS)[number];
type SortKey = "run" | "subroot" | "status" | "selected_checkpoint" | MetricKey;

function formatMetric(value: number | null | undefined): string {
  if (typeof value !== "number") {
    return "—";
  }
  if (value === 0) {
    return "0";
  }
  if (Math.abs(value) < 0.001) {
    return value.toExponential(2);
  }
  if (Math.abs(value) < 1) {
    return value.toFixed(4);
  }
  return value.toFixed(2);
}

function metricValue(metrics: CheckpointSelectionMetricMap, metric: MetricKey): number | null {
  const value = metrics[metric];
  return typeof value === "number" ? value : null;
}

function compactLabel(value: unknown): string {
  if (value == null || value === "") {
    return "—";
  }
  return String(value);
}

function stageRows(rows: CheckpointSelectionRunDetail["stage1_ranking"]): Array<Record<string, number | string>> {
  return rows.map((row) => ({
    checkpoint: compactLabel(row.checkpoint_identifier),
    rank: Number(row.rank ?? 0),
    KID: Number(row.KID ?? Number.NaN),
    FID: Number(row.FID ?? Number.NaN),
    MMD: Number(row.MMD ?? Number.NaN),
    selection_score: Number(row.selection_score ?? Number.NaN),
  }));
}

function warningsPanel(warnings: string[]) {
  if (warnings.length === 0) {
    return null;
  }
  return (
    <div className="status-panel warning">
      {warnings.slice(0, 4).map((warning) => (
        <div key={warning}>{warning}</div>
      ))}
      {warnings.length > 4 ? <div>{warnings.length - 4} more warnings</div> : null}
    </div>
  );
}

function MetricCharts({ runs }: { runs: CheckpointSelectionRunRow[] }) {
  return (
    <div className="checkpoint-chart-grid">
      {METRIC_KEYS.map((metric) => {
        const data = runs
          .map((run) => ({
            run: run.run,
            value: metricValue(run.metrics, metric),
          }))
          .filter((row) => row.value !== null);
        return (
          <div className="chart-panel" key={metric}>
            <div className="panel-header panel-header--compact">
              <div>
                <p className="eyebrow">Final Metric</p>
                <h3>{metric}</h3>
              </div>
              <p className="supporting-copy">{metric === "Intra-LPIPS" ? "higher is better" : "lower is better"}</p>
            </div>
            {data.length > 0 ? (
              <ResponsiveContainer width="100%" height={310}>
                <BarChart data={data} margin={{ left: 4, right: 18, bottom: 128 }}>
                  <CartesianGrid strokeDasharray="3 3" vertical={false} />
                  <XAxis
                    dataKey="run"
                    angle={-38}
                    textAnchor="end"
                    interval={0}
                    height={150}
                    tickMargin={12}
                  />
                  <YAxis tickFormatter={(value) => formatMetric(Number(value))} />
                  <Tooltip formatter={(value: number) => formatMetric(value)} />
                  <Bar dataKey="value" fill="#3d6cc8" radius={[6, 6, 0, 0]} />
                </BarChart>
              </ResponsiveContainer>
            ) : (
              <div className="status-panel">No {metric} values in the selected runs.</div>
            )}
          </div>
        );
      })}
    </div>
  );
}

function RunsTable({
  runs,
  selected,
  sortKey,
  sortDirection,
  onSelect,
  onSort,
}: {
  runs: CheckpointSelectionRunRow[];
  selected: CheckpointSelectionRunRow | null;
  sortKey: SortKey;
  sortDirection: "asc" | "desc";
  onSelect: (run: CheckpointSelectionRunRow) => void;
  onSort: (key: SortKey) => void;
}) {
  const sortMark = (key: SortKey) => (sortKey === key ? (sortDirection === "asc" ? " ↑" : " ↓") : "");

  return (
    <div className="group-table checkpoint-table">
      <table>
        <thead>
          <tr>
            <th><button type="button" onClick={() => onSort("subroot")}>Subroot{sortMark("subroot")}</button></th>
            <th><button type="button" onClick={() => onSort("run")}>Run{sortMark("run")}</button></th>
            <th><button type="button" onClick={() => onSort("status")}>Status{sortMark("status")}</button></th>
            <th><button type="button" onClick={() => onSort("selected_checkpoint")}>Selected{sortMark("selected_checkpoint")}</button></th>
            {METRIC_KEYS.map((metric) => (
              <th key={metric}>
                <button type="button" onClick={() => onSort(metric)}>
                  {metric}{sortMark(metric)}
                </button>
              </th>
            ))}
            <th>Previews</th>
          </tr>
        </thead>
        <tbody>
          {runs.map((run) => {
            const isSelected = selected?.relative_path === run.relative_path;
            return (
              <tr
                key={run.relative_path}
                className={isSelected ? "selected-row" : ""}
                onClick={() => onSelect(run)}
              >
                <td>{run.subroot ?? "root"}</td>
                <td><strong>{run.run}</strong></td>
                <td><span className={`status-badge status-badge--${run.status}`}>{run.status}</span></td>
                <td>{compactLabel(run.selected_checkpoint)}</td>
                {METRIC_KEYS.map((metric) => (
                  <td key={`${run.relative_path}-${metric}`}>{formatMetric(metricValue(run.metrics, metric))}</td>
                ))}
                <td>{run.available_preview_stages.join(", ") || "—"}</td>
              </tr>
            );
          })}
        </tbody>
      </table>
    </div>
  );
}

function RankingTable({ rows }: { rows: CheckpointSelectionRunDetail["stage1_ranking"] }) {
  if (rows.length === 0) {
    return <div className="status-panel">No ranking rows available.</div>;
  }
  return (
    <div className="group-table checkpoint-ranking-table">
      <table>
        <thead>
          <tr>
            <th>Rank</th>
            <th>Checkpoint</th>
            <th>KID</th>
            <th>FID</th>
            <th>MMD</th>
            <th>Score</th>
            <th>Images</th>
          </tr>
        </thead>
        <tbody>
          {rows.map((row, index) => (
            <tr key={`${compactLabel(row.checkpoint_identifier)}-${index}`}>
              <td>{compactLabel(row.rank)}</td>
              <td>{compactLabel(row.checkpoint_identifier)}</td>
              <td>{formatMetric(typeof row.KID === "number" ? row.KID : null)}</td>
              <td>{formatMetric(typeof row.FID === "number" ? row.FID : null)}</td>
              <td>{formatMetric(typeof row.MMD === "number" ? row.MMD : null)}</td>
              <td>{formatMetric(typeof row.selection_score === "number" ? row.selection_score : null)}</td>
              <td>{compactLabel(row.total_generated_images ?? row.num_generated_images)}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

function RankingChart({
  title,
  rows,
  includeFid,
}: {
  title: string;
  rows: CheckpointSelectionRunDetail["stage1_ranking"];
  includeFid: boolean;
}) {
  const data = stageRows(rows).filter((row) => Number.isFinite(row.KID));
  return (
    <div className="chart-panel">
      <div className="panel-header panel-header--compact">
        <div>
          <p className="eyebrow">Ranking</p>
          <h3>{title}</h3>
        </div>
        <p className="supporting-copy">lower is better</p>
      </div>
      {data.length > 0 ? (
        <ResponsiveContainer width="100%" height={260}>
          <LineChart data={data} margin={{ right: 20, bottom: 50 }}>
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis dataKey="checkpoint" angle={-30} textAnchor="end" interval={0} height={72} />
            <YAxis yAxisId="kid" tickFormatter={(value) => formatMetric(Number(value))} />
            <Tooltip formatter={(value: number) => formatMetric(value)} />
            <Legend />
            <Line yAxisId="kid" type="monotone" dataKey="KID" stroke="#3d6cc8" strokeWidth={2} dot={{ r: 3 }} />
            {includeFid ? (
              <Line yAxisId="kid" type="monotone" dataKey="FID" stroke="#c8553d" strokeWidth={2} dot={{ r: 3 }} />
            ) : null}
          </LineChart>
        </ResponsiveContainer>
      ) : (
        <div className="status-panel">No chartable ranking values.</div>
      )}
    </div>
  );
}

function PreviewPanel({
  root,
  preview,
}: {
  root: string;
  preview: CheckpointSelectionPreview;
}) {
  const grid = preview.preview_grid;
  return (
    <article className="checkpoint-preview">
      <div className="example-card__meta">
        <div>
          <p className="eyebrow">{preview.stage ?? "stage"}</p>
          <h4>{preview.checkpoint_identifier ?? "checkpoint"}</h4>
        </div>
        <span>{preview.num_preview_images ?? preview.preview_images.length} images</span>
      </div>
      {grid ? (
        <img
          src={getCheckpointSelectionPreviewUrl(root, grid)}
          alt={`${preview.checkpoint_identifier ?? "checkpoint"} ${preview.stage ?? "stage"} preview grid`}
          loading="lazy"
        />
      ) : (
        <div className="status-panel">No preview grid for this stage.</div>
      )}
    </article>
  );
}

function RunDetail({ detail }: { detail: CheckpointSelectionRunDetail }) {
  const finalMetricEntries = METRIC_KEYS.filter((metric) => metric in detail.metrics);
  const previews = detail.previews.filter((preview) => preview.preview_grid || preview.preview_images.length > 0);

  return (
    <section className="panel checkpoint-detail">
      <div className="detail-card__header">
        <div>
          <p className="eyebrow">Selected Run</p>
          <h2>{detail.run}</h2>
        </div>
        <div className="detail-card__stats">
          <span>{detail.subroot ?? "root"}</span>
          <span>{detail.model_type ?? "unknown model"}</span>
          <span>{detail.generation_backend_used ?? "unknown backend"}</span>
        </div>
      </div>

      {warningsPanel(detail.warnings)}

      <div className="summary-grid checkpoint-summary-grid">
        <div className="summary-card accent">
          <span>selected checkpoint</span>
          <h3>{detail.selected_checkpoint ?? "—"}</h3>
        </div>
        {finalMetricEntries.map((metric) => (
          <div className="summary-card" key={metric}>
            <span>{metric}</span>
            <h3>{formatMetric(metricValue(detail.metrics, metric))}</h3>
          </div>
        ))}
      </div>

      <div className="detail-chart-grid">
        <RankingChart title="Stage 1" rows={detail.stage1_ranking} includeFid />
        <RankingChart title="Stage 2" rows={detail.stage2_ranking} includeFid={false} />
      </div>

      <div className="detail-chart-grid">
        <div className="chart-panel">
          <div className="panel-header panel-header--compact">
            <div>
              <p className="eyebrow">Table</p>
              <h3>Stage 1 Ranking</h3>
            </div>
          </div>
          <RankingTable rows={detail.stage1_ranking} />
        </div>
        <div className="chart-panel">
          <div className="panel-header panel-header--compact">
            <div>
              <p className="eyebrow">Table</p>
              <h3>Stage 2 Ranking</h3>
            </div>
          </div>
          <RankingTable rows={detail.stage2_ranking} />
        </div>
      </div>

      <div className="panel-header checkpoint-preview-header">
        <div>
          <p className="eyebrow">Previews</p>
          <h3>Generated grids</h3>
        </div>
      </div>
      {previews.length > 0 ? (
        <div className="checkpoint-preview-grid">
          {previews.map((preview, index) => (
            <PreviewPanel
              key={`${preview.checkpoint_identifier ?? "checkpoint"}-${preview.stage ?? "stage"}-${index}`}
              root={detail.root}
              preview={preview}
            />
          ))}
        </div>
      ) : (
        <div className="status-panel">No preview images were found for this run.</div>
      )}
    </section>
  );
}

export function CheckpointSelectionView() {
  const [rootInput, setRootInput] = useState(DEFAULT_ROOT);
  const [catalog, setCatalog] = useState<CheckpointSelectionCatalogResponse | null>(null);
  const [catalogLoading, setCatalogLoading] = useState(false);
  const [catalogError, setCatalogError] = useState<string | null>(null);
  const [subrootFilter, setSubrootFilter] = useState(ALL_FILTER);
  const [runFilter, setRunFilter] = useState("");
  const [selectedRun, setSelectedRun] = useState<CheckpointSelectionRunRow | null>(null);
  const [detail, setDetail] = useState<CheckpointSelectionRunDetail | null>(null);
  const [detailLoading, setDetailLoading] = useState(false);
  const [detailError, setDetailError] = useState<string | null>(null);
  const [sortKey, setSortKey] = useState<SortKey>("run");
  const [sortDirection, setSortDirection] = useState<"asc" | "desc">("asc");

  async function loadCatalog() {
    const root = rootInput.trim();
    if (!root) {
      setCatalogError("Root path is required.");
      return;
    }
    setCatalogLoading(true);
    setCatalogError(null);
    setDetail(null);
    setSelectedRun(null);
    try {
      const payload = await getCheckpointSelectionCatalog({ root });
      setCatalog(payload);
      setSubrootFilter(ALL_FILTER);
      setRunFilter("");
      if (payload.runs.length > 0) {
        setSelectedRun(payload.runs[0]);
      }
    } catch (error) {
      setCatalog(null);
      setCatalogError(error instanceof Error ? error.message : "Failed to scan checkpoint-selection root");
    } finally {
      setCatalogLoading(false);
    }
  }

  useEffect(() => {
    void loadCatalog();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  useEffect(() => {
    if (!catalog || !selectedRun) {
      return;
    }
    let cancelled = false;
    setDetailLoading(true);
    setDetailError(null);
    getCheckpointSelectionRun({
      root: catalog.root,
      subroot: selectedRun.subroot,
      run: selectedRun.run,
    })
      .then((payload) => {
        if (!cancelled) {
          setDetail(payload);
        }
      })
      .catch((error) => {
        if (!cancelled) {
          setDetail(null);
          setDetailError(error instanceof Error ? error.message : "Failed to load run details");
        }
      })
      .finally(() => {
        if (!cancelled) {
          setDetailLoading(false);
        }
      });
    return () => {
      cancelled = true;
    };
  }, [catalog, selectedRun]);

  const filteredRuns = useMemo(() => {
    const rows = catalog?.runs ?? [];
    const filtered = rows.filter((run) => {
      const subrootMatch =
        subrootFilter === ALL_FILTER ||
        (subrootFilter === ROOT_FILTER ? run.subroot === null : run.subroot === subrootFilter);
      const runMatch = run.run.toLowerCase().includes(runFilter.trim().toLowerCase());
      return subrootMatch && runMatch;
    });
    return [...filtered].sort((left, right) => {
      const multiplier = sortDirection === "asc" ? 1 : -1;
      const leftValue = METRIC_KEYS.includes(sortKey as MetricKey)
        ? metricValue(left.metrics, sortKey as MetricKey)
        : left[sortKey as keyof CheckpointSelectionRunRow];
      const rightValue = METRIC_KEYS.includes(sortKey as MetricKey)
        ? metricValue(right.metrics, sortKey as MetricKey)
        : right[sortKey as keyof CheckpointSelectionRunRow];
      if (typeof leftValue === "number" && typeof rightValue === "number") {
        return (leftValue - rightValue) * multiplier;
      }
      return compactLabel(leftValue).localeCompare(compactLabel(rightValue)) * multiplier;
    });
  }, [catalog, runFilter, sortDirection, sortKey, subrootFilter]);

  function handleSort(key: SortKey) {
    if (key === sortKey) {
      setSortDirection((current) => (current === "asc" ? "desc" : "asc"));
      return;
    }
    setSortKey(key);
    setSortDirection(METRIC_KEYS.includes(key as MetricKey) ? "asc" : "asc");
  }

  return (
    <>
      <header className="hero checkpoint-hero">
        <div className="hero__copy">
          <p className="eyebrow">Checkpoint Selection</p>
          <h1>Browse checkpoint rankings, final metrics, and generated previews</h1>
          <p className="hero__lede">
            Compare stage-1 and stage-2 rankings across detected analysis runs, then inspect the final selected checkpoint.
          </p>
        </div>
        <div className="hero__stats">
          <div className="stat-card">
            <span>runs</span>
            <strong>{catalog?.runs.length ?? "—"}</strong>
          </div>
          <div className="stat-card">
            <span>subroots</span>
            <strong>{catalog?.subroots.length ?? "—"}</strong>
          </div>
          <div className="stat-card">
            <span>selected</span>
            <strong>{selectedRun?.selected_checkpoint ?? "—"}</strong>
          </div>
        </div>
      </header>

      <section className="panel controls-panel checkpoint-controls">
        <div className="controls-header">
          <div>
            <p className="eyebrow">Root</p>
            <h2>Analysis folder</h2>
          </div>
          <button type="button" className="primary-action" onClick={() => void loadCatalog()} disabled={catalogLoading}>
            {catalogLoading ? "Scanning..." : "Refresh"}
          </button>
        </div>
        <div className="controls-grid checkpoint-root-grid">
          <label className="controls-grid__span-2">
            <span>ROOT path</span>
            <input value={rootInput} onChange={(event) => setRootInput(event.target.value)} />
          </label>
          <label>
            <span>Subroot</span>
            <select value={subrootFilter} onChange={(event) => setSubrootFilter(event.target.value)}>
              <option value={ALL_FILTER}>All subroots</option>
              {catalog?.subroots.map((subroot) => (
                <option key={subroot ?? ROOT_FILTER} value={subroot ?? ROOT_FILTER}>
                  {subroot ?? "Root runs"}
                </option>
              ))}
            </select>
          </label>
          <label>
            <span>Run filter</span>
            <input value={runFilter} onChange={(event) => setRunFilter(event.target.value)} />
          </label>
        </div>
        {catalogError ? <div className="status-panel error">{catalogError}</div> : null}
        {catalog ? warningsPanel(catalog.warnings) : null}
      </section>

      {catalogLoading ? <div className="status-panel">Scanning checkpoint-selection root...</div> : null}

      {catalog && filteredRuns.length > 0 ? (
        <>
          <section className="panel">
            <div className="panel-header">
              <div>
                <p className="eyebrow">Overview</p>
                <h2>Detected runs</h2>
              </div>
              <p className="supporting-copy">{filteredRuns.length} visible of {catalog.runs.length} runs</p>
            </div>
            <MetricCharts runs={filteredRuns} />
            <RunsTable
              runs={filteredRuns}
              selected={selectedRun}
              sortKey={sortKey}
              sortDirection={sortDirection}
              onSelect={setSelectedRun}
              onSort={handleSort}
            />
          </section>

          {detailLoading ? <div className="status-panel">Loading selected run...</div> : null}
          {detailError ? <div className="status-panel error">{detailError}</div> : null}
          {detail ? <RunDetail detail={detail} /> : null}
        </>
      ) : null}

      {catalog && filteredRuns.length === 0 && !catalogLoading ? (
        <div className="status-panel">No checkpoint-selection runs matched the current filters.</div>
      ) : null}
    </>
  );
}
