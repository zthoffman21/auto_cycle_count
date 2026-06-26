import { useEffect, useRef, useState, type RefObject } from "react";
import { inspectImage } from "./api";
import type { InspectionCell, InspectionResponse, Point } from "./types";

function classificationProbability(cell: InspectionCell): number {
  if (cell.result === "EMPTY") return cell.emptyProbability;
  if (cell.result === "NON_EMPTY") return cell.nonEmptyProbability;
  return Math.max(cell.emptyProbability, cell.nonEmptyProbability);
}

function cellColor(result?: InspectionCell["result"]): string {
  if (result === "EMPTY") return "#8ef06a";
  if (result === "NON_EMPTY") return "#ff6b6b";
  if (result === "UNCERTAIN" || result === "BAD_CAPTURE") return "#ffc857";
  return "#5ce1e6";
}

function cellFill(result?: InspectionCell["result"]): string {
  if (result === "EMPTY") return "rgba(142,240,106,.14)";
  if (result === "NON_EMPTY") return "rgba(255,107,107,.14)";
  if (result === "UNCERTAIN" || result === "BAD_CAPTURE") return "rgba(255,200,87,.14)";
  return "rgba(92,225,230,.08)";
}

function Overlay({
  result,
  cells,
  imageRef,
}: {
  result: InspectionResponse;
  cells: InspectionCell[];
  imageRef: RefObject<HTMLImageElement>;
}) {
  const [box, setBox] = useState({ left: 0, top: 0, width: 0, height: 0 });

  useEffect(() => {
    function updateBox() {
      const image = imageRef.current;
      const stage = image?.parentElement;
      if (!image || !stage) return;
      const imageRect = image.getBoundingClientRect();
      const stageRect = stage.getBoundingClientRect();
      setBox({
        left: imageRect.left - stageRect.left,
        top: imageRect.top - stageRect.top,
        width: imageRect.width,
        height: imageRect.height,
      });
    }

    updateBox();
    window.addEventListener("resize", updateBox);
    return () => window.removeEventListener("resize", updateBox);
  }, [imageRef, result]);

  const image = imageRef.current;
  const normalized = result.coordinateSpace === "NORMALIZED_100";
  const viewBox =
    normalized || !image
      ? "0 0 100 100"
      : `0 0 ${image.naturalWidth} ${image.naturalHeight}`;
  const labelSize = normalized || !image ? 3 : Math.max(image.naturalWidth, image.naturalHeight) * 0.012;
  const classifications = new Map(cells.map((cell) => [cell.cellId, cell.result]));

  return (
    <svg
      aria-label="Computer vision detection overlay"
      className="inspection-overlay"
      style={box}
      viewBox={viewBox}
    >
      {result.totePolygon ? (
        <Polygon
          points={result.totePolygon}
          stroke="#b6f36b"
          fill="rgba(182,243,107,.12)"
          label="TOTE"
          labelSize={labelSize}
        />
      ) : null}
      {result.detectedCells.map((cell) => {
        const classification = classifications.get(cell.cellId);
        return cell.polygon.length ? (
          <Polygon
            key={cell.cellId}
            points={cell.polygon}
            stroke={cellColor(classification)}
            fill={cellFill(classification)}
            label={`${cell.cellId}${classification ? ` · ${classification}` : ""}`}
            labelSize={labelSize}
          />
        ) : null;
      })}
    </svg>
  );
}

function Polygon({
  points,
  stroke,
  fill,
  label,
  labelSize,
}: {
  points: Point[];
  stroke: string;
  fill: string;
  label: string;
  labelSize: number;
}) {
  const x = Math.min(...points.map((point) => point[0]));
  const y = Math.min(...points.map((point) => point[1]));
  return (
    <>
      <polygon
        points={points.map((point) => point.join(",")).join(" ")}
        fill={fill}
        stroke={stroke}
        strokeWidth="2"
        vectorEffect="non-scaling-stroke"
      />
      <text
        x={x + labelSize * 0.5}
        y={y + labelSize * 1.2}
        fill={stroke}
        fontSize={labelSize}
        fontFamily="monospace"
        fontWeight="700"
        paintOrder="stroke"
        stroke="#08100c"
        strokeWidth={labelSize * 0.12}
      >
        {label}
      </text>
    </>
  );
}

type BatchItem = {
  id: string;
  file: File;
  status: "loading" | "done" | "error";
  result?: InspectionResponse;
  error?: string;
};

function ResultCard({ item }: { item: BatchItem }) {
  const imageRef = useRef<HTMLImageElement>(null);
  const [imageVersion, setImageVersion] = useState(Date.now());
  const [expanded, setExpanded] = useState(false);
  const { result } = item;

  const methodNames = result?.methodResults ? Object.keys(result.methodResults) : null;
  const [selectedMethod, setSelectedMethod] = useState<string | null>(null);
  const activeCells =
    result && methodNames && selectedMethod && result.methodResults
      ? (result.methodResults[selectedMethod] ?? result.cells)
      : (result?.cells ?? []);
  const errorType =
    typeof result?.metadata?.errorType === "string" ? result.metadata.errorType : null;
  const errorMessage =
    typeof result?.metadata?.errorMessage === "string" ? result.metadata.errorMessage : null;
  const inferenceFailed = result?.reasonCode === "INFERENCE_ERROR";
  return (
    <article className="result-card panel">
      <button
        className="result-card-header"
        type="button"
        disabled={item.status === "loading"}
        onClick={() => setExpanded((e) => !e)}
        aria-expanded={expanded}
      >
        <span className="result-card-left">
          {item.status === "loading" ? (
            <span className="result-loading">
              <span className="button-spinner" />
            </span>
          ) : item.status === "error" ? (
            <span className="decision-badge error">ERROR</span>
          ) : result ? (
            <span className={`decision-badge ${result.result.toLowerCase()}`}>{result.result}</span>
          ) : null}
          <span className="result-filename">{item.file.name}</span>
          {result && (
            <span className="result-chips">
              <span className="result-chip">{result.observedLayout ?? "no detection"}</span>
              <span className="result-chip">
                {result.cells.length} cell{result.cells.length !== 1 ? "s" : ""}
              </span>
              <span className="result-chip result-chip-reason">{result.reasonCode}</span>
            </span>
          )}
          {item.status === "error" && (
            <span className="result-error-msg">{item.error}</span>
          )}
        </span>
        {item.status !== "loading" && (
          <span className="expand-chevron" aria-hidden="true">{expanded ? "▲" : "▼"}</span>
        )}
      </button>

      {expanded && result && (
        <div className="result-detail">
          <article className="panel viewer-panel">
            <div className="panel-heading compact">
              <div>
                <p className="step-label">visual evidence</p>
                <h2>Detection overlay</h2>
              </div>
              <div className="legend">
                <span>
                  <i className="tote-key" /> tote ROI
                </span>
                <span>
                  <i className="cell-key" /> cells
                </span>
              </div>
            </div>
            <div className="image-stage">
              <img
                ref={imageRef}
                alt="Uploaded tote inspection"
                onLoad={() => setImageVersion(Date.now())}
                src={result.imageUri}
              />
              <Overlay key={imageVersion} result={result} cells={activeCells} imageRef={imageRef} />
            </div>
          </article>

          <aside className="results-column">
            <article className="panel decision-panel">
              <p className="step-label">prediction</p>
              <div className="decision-header">
                <div>
                  <h2>{result.observedLayout || "NO DETECTION"}</h2>
                </div>
                <div
                  className="confidence-ring"
                  style={{
                    background: `conic-gradient(var(--accent) ${Math.round(
                      (result.layoutConfidence ?? 0) * 100,
                    )}%, #29312d 0)`,
                  }}
                >
                  <span>{Math.round((result.layoutConfidence ?? 0) * 100)}%</span>
                </div>
              </div>
              <dl className="result-meta">
                <div>
                  <dt>Inspection</dt>
                  <dd>{result.inspectionId}</dd>
                </div>
                <div>
                  <dt>Decision</dt>
                  <dd>{result.result}</dd>
                </div>
                <div>
                  <dt>Reason</dt>
                  <dd>{result.reasonCode}</dd>
                </div>
                <div>
                  <dt>Detected cells</dt>
                  <dd>{result.detectedCells.length}</dd>
                </div>
                <div>
                  <dt>Geometry</dt>
                  <dd title={result.geometry?.issues.join("; ")}>
                    {result.geometry?.valid ? "valid" : "invalid"}
                  </dd>
                </div>
              </dl>
              {result.geometry && !result.geometry.valid && result.geometry.issues.length ? (
                <ul className="geometry-issues">
                  {result.geometry.issues.map((issue) => <li key={issue}>{issue}</li>)}
                </ul>
              ) : null}
              {inferenceFailed && (errorType || errorMessage) ? (
                <ul className="geometry-issues">
                  <li>{[errorType, errorMessage].filter(Boolean).join(": ")}</li>
                </ul>
              ) : null}
            </article>

            <article className="panel cells-panel">
              <div className="panel-heading compact">
                <div>
                  <p className="step-label">cell results</p>
                  <h2>Per-cell output</h2>
                </div>
                {methodNames && methodNames.length > 1 && (
                  <div className="method-toggle" role="group" aria-label="Classifier method">
                    {methodNames.map((method) => (
                      <button
                        key={method}
                        type="button"
                        className={`method-toggle-btn${selectedMethod === method ? " active" : ""}`}
                        onClick={() => setSelectedMethod(selectedMethod === method ? null : method)}
                      >
                        {method}
                      </button>
                    ))}
                  </div>
                )}
              </div>
              <div className="cell-results">
                {activeCells.length ? (
                  activeCells.map((cell) => {
                    const probability = classificationProbability(cell);
                    return (
                      <div className={`cell-row state-${cell.result.toLowerCase()}`} key={cell.cellId}>
                        <span className="cell-id">{cell.cellId}</span>
                        <div>
                          <span className="cell-state">{cell.result}</span>
                          <div className="meter">
                            <i
                              style={{
                                width: `${probability * 100}%`,
                                backgroundColor: cellColor(cell.result),
                              }}
                            />
                          </div>
                          <small className="cell-probabilities">
                            E {Math.round(cell.emptyProbability * 100)}% · N{" "}
                            {Math.round(cell.nonEmptyProbability * 100)}%
                          </small>
                        </div>
                        <span className="probability">{Math.round(probability * 100)}%</span>
                      </div>
                    );
                  })
                ) : (
                  <p className="probability">No cell predictions produced.</p>
                )}
              </div>
            </article>

            <article className="panel pipeline-panel">
              <p className="step-label">inference trace</p>
              <ol>
                {[
                  ["Image accepted", result.imageUri],
                  [
                    "Tote ROI detection",
                    result.modelVersions.toteDetector || (inferenceFailed ? "failed" : "not run"),
                  ],
                  ["Cell layout detection", result.modelVersions.layoutDetector || "not run"],
                  [
                    "Geometry validation",
                    result.geometry ? (result.geometry.valid ? "valid" : "invalid") : "not run",
                  ],
                  ["Cell classification", result.modelVersions.cellClassifier || "not run"],
                  [
                    "Prediction assembled",
                    inferenceFailed && errorType
                      ? `failed: ${errorType}`
                      : `${result.detectedCells.length} cells`,
                  ],
                ].map(([name, detail]) => (
                  <li key={name}>
                    {name}
                    <small>{detail}</small>
                  </li>
                ))}
              </ol>
            </article>
          </aside>
        </div>
      )}
    </article>
  );
}

export function InspectionPage() {
  const [files, setFiles] = useState<File[]>([]);
  const [dragging, setDragging] = useState(false);
  const [globalError, setGlobalError] = useState("");
  const [batchItems, setBatchItems] = useState<BatchItem[]>([]);

  const submitting = batchItems.some((b) => b.status === "loading");
  const doneCount = batchItems.filter((b) => b.status !== "loading").length;

  function selectFiles(fileList: FileList | null) {
    if (fileList && fileList.length > 0) setFiles(Array.from(fileList));
  }

  async function handleSubmit(event: React.FormEvent<HTMLFormElement>) {
    event.preventDefault();
    if (!files.length) return;

    setGlobalError("");

    const newItems: BatchItem[] = files.map((file, i) => ({
      id: `${Date.now()}-${i}`,
      file,
      status: "loading",
    }));
    setBatchItems(newItems);

    newItems.forEach((item) => {
      const fd = new FormData();
      fd.append("image", item.file);
      inspectImage(fd)
        .then((result) => {
          setBatchItems((prev) =>
            prev.map((b) => (b.id === item.id ? { ...b, status: "done", result } : b)),
          );
        })
        .catch((err: unknown) => {
          setBatchItems((prev) =>
            prev.map((b) =>
              b.id === item.id
                ? {
                    ...b,
                    status: "error",
                    error: err instanceof Error ? err.message : "Inspection failed",
                  }
                : b,
            ),
          );
        });
    });
  }

  const fileLabel =
    files.length === 0
      ? ""
      : files.length === 1
        ? files[0].name
        : `${files.length} images selected`;

  return (
    <>
      <header className="topbar">
        <div className="top-actions">
          <a href="/train">Training data</a>
        </div>
      </header>

      <main className="page-shell">
        <section className="panel input-panel">
          <div className="panel-heading">
            <div>
              <p className="step-label">01 / inspection input</p>
              <h2>Submit tote image{files.length !== 1 ? "s" : ""}</h2>
            </div>
            <p>Manual stand-in for scanner and camera events.</p>
          </div>

          <form onSubmit={handleSubmit}>
            <label
              className={`drop-zone ${dragging ? "dragging" : ""}`}
              onDragEnter={(e) => {
                e.preventDefault();
                setDragging(true);
              }}
              onDragOver={(e) => {
                e.preventDefault();
                setDragging(true);
              }}
              onDragLeave={(e) => {
                e.preventDefault();
                setDragging(false);
              }}
              onDrop={(e) => {
                e.preventDefault();
                setDragging(false);
                selectFiles(e.dataTransfer.files);
              }}
            >
              <input
                id="image"
                name="image"
                type="file"
                accept="image/jpeg,image/png,image/webp"
                multiple
                required
                onChange={(e) => selectFiles(e.target.files)}
              />
              <span className="drop-icon">+</span>
              <strong>Drop tote images here</strong>
              <small>JPEG, PNG, or WebP · multiple files supported</small>
              <span className="file-name">{fileLabel}</span>
            </label>

            <button type="submit" disabled={submitting}>
              <span>
                {submitting
                  ? `Inspecting… ${doneCount} / ${batchItems.length}`
                  : "Run empty-tote inspection"}
              </span>
              <span aria-hidden="true">→</span>
            </button>
            <p className="form-error" role="alert">
              {globalError}
            </p>
          </form>
        </section>

        {batchItems.length > 0 && (
          <section className="results-section">
            <div className="results-section-header">
              <p className="step-label">02 / inspection results</p>
              <h2>
                {submitting
                  ? `Results · ${doneCount} of ${batchItems.length} complete`
                  : `Results · ${batchItems.length} image${batchItems.length !== 1 ? "s" : ""}`}
              </h2>
            </div>
            <div className="results-list">
              {batchItems.map((item) => (
                <ResultCard key={item.id} item={item} />
              ))}
            </div>
          </section>
        )}
      </main>
    </>
  );
}
