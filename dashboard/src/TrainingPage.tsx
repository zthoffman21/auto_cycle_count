import { useEffect, useMemo, useRef, useState } from "react";
import {
  deleteAllTrainingImages,
  deleteTrainingImage,
  getTrainingImage,
  getTrainingStatus,
  listTrainingImages,
  updateTrainingImage,
  uploadTrainingImages,
} from "./api";
import type { CellState, DatasetSplit, Point, ToteLayout, TrainingImage, TrainingRegion } from "./types";

type ToolMode = "idle" | "tote-polygon" | "tote-box" | "divider-line";

interface DraftBox {
  start: Point;
  current: Point;
}

interface LabelOffset {
  xPercent: number;
  yPercent: number;
}

interface DraggingLabel {
  regionId: string;
  startClientX: number;
  startClientY: number;
  startOffset: LabelOffset;
  moved: boolean;
}

function pointsMatch(a: Point[], b: Point[]) {
  return JSON.stringify(a) === JSON.stringify(b);
}

function polygonCenter(polygon: Point[]): Point {
  return [
    polygon.reduce((total, point) => total + point[0], 0) / polygon.length,
    polygon.reduce((total, point) => total + point[1], 0) / polygon.length,
  ];
}

function signedDistance(point: Point, lineStart: Point, lineEnd: Point): number {
  return (
    (lineEnd[0] - lineStart[0]) * (point[1] - lineStart[1]) -
    (lineEnd[1] - lineStart[1]) * (point[0] - lineStart[0])
  );
}

function clipPolygonToLine(
  polygon: Point[],
  lineStart: Point,
  lineEnd: Point,
  keepPositive: boolean,
): Point[] {
  const result: Point[] = [];
  for (let index = 0; index < polygon.length; index += 1) {
    const current = polygon[index];
    const previous = polygon[(index + polygon.length - 1) % polygon.length];
    const currentDistance = signedDistance(current, lineStart, lineEnd);
    const previousDistance = signedDistance(previous, lineStart, lineEnd);
    const currentInside = keepPositive ? currentDistance >= 0 : currentDistance <= 0;
    const previousInside = keepPositive ? previousDistance >= 0 : previousDistance <= 0;
    if (currentInside !== previousInside) {
      const ratio = previousDistance / (previousDistance - currentDistance);
      result.push([
        previous[0] + ratio * (current[0] - previous[0]),
        previous[1] + ratio * (current[1] - previous[1]),
      ]);
    }
    if (currentInside) result.push(current);
  }
  return result;
}

function splitPolygonByLine(polygon: Point[], [lineStart, lineEnd]: Point[]): [Point[], Point[]] {
  return [
    clipPolygonToLine(polygon, lineStart, lineEnd, true),
    clipPolygonToLine(polygon, lineStart, lineEnd, false),
  ];
}

function generateCellPolygons(totePolygon: Point[], dividers: TrainingRegion[]): Point[][] | null {
  let polygons = [totePolygon];
  for (const divider of dividers) {
    const next: Point[][] = [];
    for (const polygon of polygons) {
      const [first, second] = splitPolygonByLine(polygon, divider.polygon);
      if (first.length < 3 || second.length < 3) return null;
      next.push(first, second);
    }
    polygons = next;
  }
  return polygons;
}

function orderCellPolygons(polygons: Point[][]): Point[][] {
  const entries = polygons.map((polygon) => ({ polygon, center: polygonCenter(polygon) }));
  if (entries.length === 2) {
    const xSpread = Math.abs(entries[0].center[0] - entries[1].center[0]);
    const ySpread = Math.abs(entries[0].center[1] - entries[1].center[1]);
    entries.sort((a, b) =>
      xSpread >= ySpread ? a.center[0] - b.center[0] : a.center[1] - b.center[1],
    );
    return entries.map((entry) => entry.polygon);
  }
  entries.sort((a, b) => a.center[1] - b.center[1]);
  const top = entries.slice(0, 2).sort((a, b) => a.center[0] - b.center[0]);
  const bottom = entries.slice(2).sort((a, b) => a.center[0] - b.center[0]);
  return [...top, ...bottom].map((entry) => entry.polygon);
}

function expectedDividers(layout: ToteLayout): number {
  if (layout === "TWO_CELL") return 1;
  if (layout === "FOUR_CELL") return 2;
  return 0;
}

function expectedCells(layout: ToteLayout): number {
  if (layout === "OPEN") return 1;
  if (layout === "TWO_CELL") return 2;
  if (layout === "FOUR_CELL") return 4;
  return 0;
}

function regionLabel(layout: ToteLayout, region: TrainingRegion): string {
  if (region.regionClass === "tote") {
    return layout === "OPEN" ? "TOTE + CELL A" : "TOTE";
  }
  return `CELL ${region.cellId || "?"}`;
}

function regionLabelPosition(region: TrainingRegion): Point {
  if (region.regionClass === "cell") {
    return polygonCenter(region.polygon);
  }
  return [
    Math.min(...region.polygon.map((point) => point[0])) + 8,
    Math.min(...region.polygon.map((point) => point[1])) + 18,
  ];
}

function cellStateClass(cellState: CellState | null): string {
  if (cellState === "EMPTY") return "state-empty";
  if (cellState === "NON_EMPTY") return "state-non-empty";
  return "state-unlabeled";
}

function nextCellState(cellState: CellState | null): CellState {
  if (cellState === null) return "EMPTY";
  if (cellState === "EMPTY") return "NON_EMPTY";
  return "EMPTY";
}

export function TrainingPage() {
  const [images, setImages] = useState<TrainingImage[]>([]);
  const [current, setCurrent] = useState<TrainingImage | null>(null);
  const [regions, setRegions] = useState<TrainingRegion[]>([]);
  const [selectedRegionId, setSelectedRegionId] = useState<string | null>(null);
  const [shape, setShape] = useState<"polygon" | "box">("polygon");
  const [tool, setTool] = useState<ToolMode>("idle");
  const [draftPoints, setDraftPoints] = useState<Point[]>([]);
  const [draftBox, setDraftBox] = useState<DraftBox | null>(null);
  const [dirty, setDirty] = useState(false);
  const [status, setStatus] = useState({
    readyImages: 0,
    draftImages: 0,
    exportedCellCrops: 0,
    exportInProgress: false,
    exportPending: false,
    exportError: null as string | null,
  });
  const [saveState, setSaveState] = useState("—");
  const [error, setError] = useState("");
  const [progress, setProgress] = useState("");
  const [split, setSplit] = useState<DatasetSplit>("train");
  const [layout, setLayout] = useState<ToteLayout>("UNKNOWN");
  const [saving, setSaving] = useState(false);
  const [labelOffsets, setLabelOffsets] = useState<Record<string, LabelOffset>>({});
  const [draggingLabelId, setDraggingLabelId] = useState<string | null>(null);
  const imageRef = useRef<HTMLImageElement>(null);
  const layerRef = useRef<SVGSVGElement>(null);
  const draggingLabelRef = useRef<DraggingLabel | null>(null);
  const suppressLabelClickRef = useRef(false);

  useEffect(() => {
    void refreshData();
  }, []);

  useEffect(() => {
    function handlePointerMove(event: PointerEvent) {
      const drag = draggingLabelRef.current;
      const image = imageRef.current;
      if (!drag || !image) return;
      const width = image.clientWidth;
      const height = image.clientHeight;
      if (!width || !height) return;
      if (
        !drag.moved &&
        (Math.abs(event.clientX - drag.startClientX) > 3 ||
          Math.abs(event.clientY - drag.startClientY) > 3)
      ) {
        drag.moved = true;
        suppressLabelClickRef.current = true;
      }
      const deltaX = ((event.clientX - drag.startClientX) / width) * 100;
      const deltaY = ((event.clientY - drag.startClientY) / height) * 100;
      setLabelOffsets((currentOffsets) => ({
        ...currentOffsets,
        [drag.regionId]: {
          xPercent: drag.startOffset.xPercent + deltaX,
          yPercent: drag.startOffset.yPercent + deltaY,
        },
      }));
    }

    function handlePointerUp() {
      draggingLabelRef.current = null;
      setDraggingLabelId(null);
      window.setTimeout(() => {
        suppressLabelClickRef.current = false;
      }, 0);
    }

    window.addEventListener("pointermove", handlePointerMove);
    window.addEventListener("pointerup", handlePointerUp);
    return () => {
      window.removeEventListener("pointermove", handlePointerMove);
      window.removeEventListener("pointerup", handlePointerUp);
    };
  }, []);

  async function refreshData(selectImageId?: string) {
    const [listedImages, trainingStatus] = await Promise.all([listTrainingImages(), getTrainingStatus()]);
    setImages(listedImages);
    setStatus(trainingStatus);
    if (selectImageId) {
      const selected = listedImages.find((item) => item.imageId === selectImageId);
      if (selected) {
        await selectImage(selected.imageId, false);
      }
    }
  }

  async function selectImage(imageId: string, promptOnDirty = true) {
    if (dirty && promptOnDirty && !window.confirm("Discard unsaved annotation changes?")) return;
    const image = await getTrainingImage(imageId);
    const nextLayout = image.layout;
    const synchronized = synchronizeOpenArea(nextLayout, image.regions);
    setCurrent(image);
    setRegions(synchronized.regions);
    setSelectedRegionId(null);
    setSplit(image.split);
    setLayout(nextLayout);
    setDraftPoints([]);
    setDraftBox(null);
    setTool("idle");
    setLabelOffsets({});
    setDraggingLabelId(null);
    setDirty(synchronized.changed);
    setSaveState(synchronized.changed ? "dirty" : "saved");
    setError("");
  }

  function synchronizeOpenArea(
    nextLayout: ToteLayout,
    sourceRegions: TrainingRegion[],
  ): { regions: TrainingRegion[]; changed: boolean } {
    if (nextLayout !== "OPEN") return { regions: sourceRegions, changed: false };
    const tote = sourceRegions.find((region) => region.regionClass === "tote");
    const cell = sourceRegions.find((region) => region.regionClass === "cell");
    const source = tote || cell;
    if (!source) return { regions: sourceRegions, changed: false };
    const synchronized =
      tote &&
      cell &&
      cell.cellId === "A" &&
      pointsMatch(tote.polygon, cell.polygon) &&
      sourceRegions.filter((region) => region.regionClass === "tote").length === 1 &&
      sourceRegions.filter((region) => region.regionClass === "cell").length === 1;
    if (synchronized) return { regions: sourceRegions, changed: false };
    return {
      regions: [
        ...sourceRegions.filter(
          (region) =>
            region.regionClass !== "tote" &&
            region.regionClass !== "cell" &&
            region.regionClass !== "divider",
        ),
        {
          regionId: tote?.regionId ?? crypto.randomUUID(),
          regionClass: "tote" as const,
          polygon: structuredClone(source.polygon),
          cellId: null,
          cellState: null,
        },
        {
          regionId: cell?.regionId ?? crypto.randomUUID(),
          regionClass: "cell" as const,
          polygon: structuredClone(source.polygon),
          cellId: "A",
          cellState: cell?.cellState ?? null,
        },
      ],
      changed: true,
    };
  }

  function markDirty() {
    setDirty(true);
    setSaveState("dirty");
  }

  const wizard = useMemo(() => {
    const tote = regions.find((region) => region.regionClass === "tote");
    const dividers = regions.filter((region) => region.regionClass === "divider");
    const cells = regions.filter((region) => region.regionClass === "cell");
    const dividerTarget = expectedDividers(layout);
    const cellTarget = expectedCells(layout);
    const layoutSelected = cellTarget > 0;
    const toteComplete = Boolean(tote);
    const dividersComplete = layoutSelected && dividers.length === dividerTarget;
    const cellsComplete = cells.length === cellTarget;
    const labelsComplete = cellsComplete && cells.every((cell) => cell.cellState);
    const complete =
      cellTarget > 0 &&
      regions.filter((region) => region.regionClass === "tote").length === 1 &&
      dividers.length === dividerTarget &&
      cells.length === cellTarget &&
      cells.every((cell) => cell.cellState);
    return {
      tote,
      dividers,
      cells,
      dividerTarget,
      cellTarget,
      layoutSelected,
      toteComplete,
      dividersComplete,
      cellsComplete,
      labelsComplete,
      complete,
    };
  }, [layout, regions]);

  function clearDraft() {
    setDraftPoints([]);
    setDraftBox(null);
  }

  function selectTool(next: "tote" | "divider") {
    setTool(next === "divider" ? "divider-line" : shape === "polygon" ? "tote-polygon" : "tote-box");
    clearDraft();
  }

  function setOpenArea(polygon: Point[]) {
    setRegions((currentRegions) => {
      const existingCell = currentRegions.find((region) => region.regionClass === "cell");
      const tote = currentRegions.find((region) => region.regionClass === "tote");
      return [
        ...currentRegions.filter(
          (region) =>
            region.regionClass !== "tote" &&
            region.regionClass !== "cell" &&
            region.regionClass !== "divider",
        ),
        {
          regionId: tote?.regionId ?? crypto.randomUUID(),
          regionClass: "tote" as const,
          polygon: structuredClone(polygon),
          cellId: null,
          cellState: null,
        },
        {
          regionId: existingCell?.regionId ?? crypto.randomUUID(),
          regionClass: "cell" as const,
          polygon: structuredClone(polygon),
          cellId: "A",
          cellState: existingCell?.cellState ?? null,
        },
      ];
    });
    setSelectedRegionId(null);
    markDirty();
  }

  function addRegion(regionClass: "tote", polygon: Point[]) {
    if (layout === "OPEN") {
      setOpenArea(polygon);
      setTool("idle");
      clearDraft();
      return;
    }
    setRegions([
      {
        regionId: crypto.randomUUID(),
        regionClass,
        polygon,
        cellId: null,
        cellState: null,
      },
    ]);
    setSelectedRegionId(null);
    setTool("idle");
    clearDraft();
    markDirty();
  }

  function addDividerLine(points: Point[]) {
    const tote = wizard.tote;
    if (!tote || wizard.dividers.length >= wizard.dividerTarget) return;
    const divider: TrainingRegion = {
      regionId: crypto.randomUUID(),
      regionClass: "divider" as const,
      polygon: points,
      cellId: null,
      cellState: null,
    };
    const generated = generateCellPolygons(tote.polygon, [...wizard.dividers, divider]);
    if (!generated) {
      setError("Divider must cross the full tote area. Draw it from edge to edge.");
      return;
    }
    const oldStates = new Map(
      regions
        .filter((region) => region.regionClass === "cell")
        .map((region) => [region.cellId, region.cellState] as const),
    );
    const geometryComplete = wizard.dividers.length + 1 === wizard.dividerTarget;
    const cells = geometryComplete
      ? orderCellPolygons(generated).map((polygon, index) => {
          const cellId = String.fromCharCode(65 + index);
          return {
            regionId: crypto.randomUUID(),
            regionClass: "cell" as const,
            polygon,
            cellId,
            cellState: oldStates.get(cellId) ?? null,
          };
        })
      : [];
    setRegions([
      ...regions.filter(
        (region) => region.regionClass !== "cell" && region.regionClass !== "divider",
      ),
      ...wizard.dividers,
      divider,
      ...cells,
    ]);
    setTool("idle");
    clearDraft();
    setError("");
    markDirty();
  }

  function canvasPoint(event: React.PointerEvent<SVGSVGElement> | React.MouseEvent<SVGSVGElement>): Point {
    const svg = layerRef.current;
    if (!svg || !current) return [0, 0];
    const point = svg.createSVGPoint();
    point.x = "clientX" in event ? event.clientX : 0;
    point.y = "clientY" in event ? event.clientY : 0;
    const transformed = point.matrixTransform(svg.getScreenCTM()?.inverse());
    return [
      Math.max(0, Math.min(current.width, transformed.x)),
      Math.max(0, Math.min(current.height, transformed.y)),
    ];
  }

  function handleCanvasClick(event: React.MouseEvent<SVGSVGElement>) {
    if (!current || tool !== "tote-polygon" || event.detail > 1) return;
    setDraftPoints((points) => [...points, canvasPoint(event)]);
  }

  function handleCanvasDoubleClick(event: React.MouseEvent<SVGSVGElement>) {
    if (tool !== "tote-polygon") return;
    event.preventDefault();
    if (draftPoints.length >= 3) addRegion("tote", draftPoints);
  }

  function handlePointerDown(event: React.PointerEvent<SVGSVGElement>) {
    if (!current || (tool !== "tote-box" && tool !== "divider-line")) return;
    const point = canvasPoint(event);
    setDraftBox({ start: point, current: point });
    event.currentTarget.setPointerCapture(event.pointerId);
  }

  function handlePointerMove(event: React.PointerEvent<SVGSVGElement>) {
    if (!draftBox) return;
    setDraftBox({ ...draftBox, current: canvasPoint(event) });
  }

  function handlePointerUp(event: React.PointerEvent<SVGSVGElement>) {
    if (!draftBox) return;
    const end = canvasPoint(event);
    const { start } = draftBox;
    setDraftBox(null);
    if (tool === "divider-line") {
      if (Math.hypot(end[0] - start[0], end[1] - start[1]) >= 5) addDividerLine([start, end]);
      return;
    }
    if (Math.abs(end[0] - start[0]) < 3 || Math.abs(end[1] - start[1]) < 3) return;
    addRegion("tote", [
      [start[0], start[1]],
      [end[0], start[1]],
      [end[0], end[1]],
      [start[0], end[1]],
    ]);
  }

  useEffect(() => {
    function handleEscape(event: KeyboardEvent) {
      if (event.key === "Escape") clearDraft();
    }
    window.addEventListener("keydown", handleEscape);
    return () => window.removeEventListener("keydown", handleEscape);
  });

  async function handleUpload(event: React.ChangeEvent<HTMLInputElement>) {
    const files = event.target.files;
    if (!files?.length) return;
    setProgress(`Uploading ${files.length} image(s)...`);
    const formData = new FormData();
    Array.from(files).forEach((file) => formData.append("images", file));
    try {
      const payload = await uploadTrainingImages(formData);
      setProgress(`${payload.length} image(s) uploaded`);
      await refreshData(payload[0]?.imageId);
    } catch (uploadError) {
      setProgress(uploadError instanceof Error ? uploadError.message : "Upload failed");
    } finally {
      event.target.value = "";
    }
  }

  async function handleSave() {
    if (!current) return;
    setError("");
    setSaving(true);
    const currentIndex = images.findIndex((image) => image.imageId === current.imageId);
    const nextImageId =
      currentIndex >= 0 && currentIndex < images.length - 1
        ? images[currentIndex + 1].imageId
        : current.imageId;
    try {
      const saved = await updateTrainingImage(current.imageId, {
        split,
        layout,
        regions,
        ready: wizard.complete,
      });
      setCurrent(saved);
      setRegions(saved.regions);
      setDirty(false);
      setSaveState("saved");
      await refreshData(wizard.complete ? nextImageId : saved.imageId);
    } catch (saveError) {
      setError(saveError instanceof Error ? saveError.message : "Save failed");
      setSaveState("dirty");
    } finally {
      setSaving(false);
    }
  }

  async function handleDeleteSelected() {
    if (!current) return;
    if (
      !window.confirm(
        `Delete ${current.originalFilename} and remove it from generated training data?`,
      )
    ) {
      return;
    }
    await deleteTrainingImage(current.imageId);
    clearCurrentImage();
    await refreshData();
  }

  async function handleDeleteAll() {
    if (!images.length) return;
    if (
      !window.confirm(
        `Delete all ${images.length} uploaded images, annotations, and generated exports?`,
      )
    ) {
      return;
    }
    await deleteAllTrainingImages();
    clearCurrentImage();
    await refreshData();
  }

  function clearCurrentImage() {
    setCurrent(null);
    setRegions([]);
    setSelectedRegionId(null);
    setDirty(false);
    setSplit("train");
    setLayout("UNKNOWN");
    setLabelOffsets({});
    setDraggingLabelId(null);
    setSaveState("saved");
    setError("");
    clearDraft();
  }

  function handleLayoutChange(nextLayout: ToteLayout) {
    setLayout(nextLayout);
    setRegions([]);
    setSelectedRegionId(null);
    setTool("idle");
    setLabelOffsets({});
    clearDraft();
    markDirty();
  }

  function resetGeometry() {
    if (regions.length && !window.confirm("Clear the tote, divider, and generated cell geometry?")) {
      return;
    }
    setRegions([]);
    setSelectedRegionId(null);
    setTool("idle");
    clearDraft();
    markDirty();
  }

  function advanceCellState(regionId: string) {
    setRegions((currentRegions) =>
      currentRegions.map((region) =>
        region.regionId === regionId
          ? { ...region, cellState: nextCellState(region.cellState) }
          : region,
      ),
    );
    markDirty();
  }

  const draftShape = useMemo(() => {
    if (!draftBox) return draftPoints;
    if (tool === "divider-line") return [draftBox.start, draftBox.current];
    const [x1, y1] = draftBox.start;
    const [x2, y2] = draftBox.current;
    return [
      [x1, y1],
      [x2, y1],
      [x2, y2],
      [x1, y2],
      [x1, y1],
    ] as Point[];
  }, [draftBox, draftPoints, tool]);

  const guideText = !wizard.layoutSelected
    ? {
        title: "Step 1: select the tote layout",
        body: "Choose Open, Two cell, or Four cell in the form.",
      }
    : !wizard.toteComplete
      ? {
          title: "Step 2: outline the tote",
          body:
            shape === "polygon"
              ? "Click around the tote boundary, then double-click the final point."
              : "Drag from one tote corner to the opposite corner.",
        }
      : !wizard.dividersComplete
        ? {
            title: `Step 3: mark divider ${wizard.dividers.length + 1} of ${wizard.dividerTarget}`,
            body: "Drag a line across the center of the divider from tote edge to tote edge.",
          }
        : !wizard.labelsComplete
          ? {
              title: "Step 4: label each generated cell",
              body: "Use the cell label fields in the form, then save.",
            }
          : {
              title: "Annotation complete and ready to export",
              body: "Review the generated cell boundaries, then save and export.",
            };

  return (
    <>
      <header className="topbar">
        <div>
          <p className="eyebrow">WES engineering tools</p>
          <h1>Training data studio</h1>
        </div>
        <nav className="page-nav">
          <a href="/">Inspection console</a>
          <a className="active" href="/train">
            Training data
          </a>
        </nav>
      </header>

      <main className="training-shell">
        <aside className="panel image-sidebar">
          <div className="sidebar-heading">
            <div>
              <p className="step-label">dataset</p>
              <h2>Source images</h2>
            </div>
            <span>{images.length}</span>
          </div>
          <label className="upload-button">
            <input
              type="file"
              accept="image/jpeg,image/png,image/webp"
              multiple
              onChange={handleUpload}
            />
            <span>+ Upload images · 50 MB each</span>
          </label>
          <div className="deletion-actions">
            <button type="button" disabled={!current} onClick={() => void handleDeleteSelected()}>
              Delete selected
            </button>
            <button type="button" disabled={!images.length} onClick={() => void handleDeleteAll()}>
              Delete all
            </button>
          </div>
          <div className="upload-progress">{progress}</div>
          <div className="image-list">
            {images.map((image) => (
              <button
                key={image.imageId}
                className={`image-item ${current?.imageId === image.imageId ? "active" : ""}`}
                onClick={() => void selectImage(image.imageId)}
                type="button"
              >
                <img alt="" src={image.imageUri} />
                <div>
                  <strong>{image.originalFilename}</strong>
                  <small>
                    {image.layout} · {image.split}
                  </small>
                </div>
                <i className={`ready-dot ${image.ready ? "ready" : ""}`} />
              </button>
            ))}
          </div>
          <div className="dataset-status">
            <p className="step-label">export status</p>
            {status.exportInProgress || status.exportPending ? (
              <p className="export-note">Export is rebuilding in the background.</p>
            ) : null}
            {status.exportError ? <p className="export-note error">{status.exportError}</p> : null}
            <dl>
              <div>
                <dt>Ready</dt>
                <dd>{status.readyImages}</dd>
              </div>
              <div>
                <dt>Drafts</dt>
                <dd>{status.draftImages}</dd>
              </div>
              <div>
                <dt>Cell crops</dt>
                <dd>{status.exportedCellCrops}</dd>
              </div>
            </dl>
          </div>
        </aside>

        <section className="panel annotation-workspace">
          <div className="annotation-toolbar">
            <span className="guide-status">{guideText.title}</span>
            <button type="button" disabled={!regions.length} onClick={resetGeometry}>
              Reset geometry
            </button>
          </div>
          {current ? (
            <div className="annotation-viewport">
              <div className="annotation-frame">
                <img
                  ref={imageRef}
                  alt="Training annotation source"
                  src={current.imageUri}
                />
                <svg
                  ref={layerRef}
                  className={tool === "idle" ? "select-mode" : ""}
                  onClick={handleCanvasClick}
                  onDoubleClick={handleCanvasDoubleClick}
                  onPointerDown={handlePointerDown}
                  onPointerMove={handlePointerMove}
                  onPointerUp={handlePointerUp}
                  viewBox={`0 0 ${current.width} ${current.height}`}
                >
                  {regions.map((region) => {
                    const openCellHidden =
                      region.regionClass === "cell" &&
                      layout === "OPEN" &&
                      regions.some(
                        (candidate) =>
                          candidate.regionClass === "tote" &&
                          pointsMatch(candidate.polygon, region.polygon),
                      );
                    if (openCellHidden) return null;
                    return region.regionClass === "divider" ? (
                      <line
                        key={region.regionId}
                        className={`annotation-region divider ${
                          region.regionId === selectedRegionId ? "selected" : ""
                        }`}
                        x1={region.polygon[0][0]}
                        x2={region.polygon[1][0]}
                        y1={region.polygon[0][1]}
                        y2={region.polygon[1][1]}
                      />
                    ) : (
                      <g key={region.regionId}>
                        {(() => {
                          const [labelX, labelY] = regionLabelPosition(region);
                          const centered = region.regionClass === "cell";
                          return (
                            <>
                        <polygon
                          className={`annotation-region ${region.regionClass} ${
                            region.regionClass === "cell" ? cellStateClass(region.cellState) : ""
                          } ${region.regionId === selectedRegionId ? "selected" : ""}`}
                          onClick={
                            region.regionClass === "cell"
                              ? (event) => {
                                  event.stopPropagation();
                                  advanceCellState(region.regionId);
                                }
                              : undefined
                          }
                          points={region.polygon.map((point) => point.join(",")).join(" ")}
                        />
                        {region.regionClass !== "cell" ? (
                          <text
                            className="region-label"
                            fill={region.regionClass === "tote" ? "#b6f36b" : "#5ce1e6"}
                            textAnchor={centered ? "middle" : "start"}
                            x={labelX}
                            y={labelY}
                          >
                            {regionLabel(layout, region)}
                          </text>
                        ) : null}
                            </>
                          );
                        })()}
                      </g>
                    );
                  })}
                  {draftShape.length ? (
                    tool === "divider-line" ? (
                      <polyline
                        className="draft-region"
                        fill="none"
                        points={draftShape.map((point) => point.join(",")).join(" ")}
                      />
                    ) : (
                      <polyline
                        className="draft-region"
                        points={draftShape.map((point) => point.join(",")).join(" ")}
                      />
                    )
                  ) : null}
                </svg>
                <div className="region-label-layer" aria-hidden="true">
                  {regions
                    .filter((region) => region.regionClass === "cell")
                    .map((region) => {
                      const [labelX, labelY] = regionLabelPosition(region);
                      const offset = labelOffsets[region.regionId] ?? {
                        xPercent: 0,
                        yPercent: 0,
                      };
                      return (
                        <div
                          className={`region-label-pill ${
                            cellStateClass(region.cellState)
                          } ${
                            draggingLabelId === region.regionId ? "dragging" : ""
                          }`}
                          key={`label-${region.regionId}`}
                          onPointerDown={(event) => {
                            event.preventDefault();
                            event.stopPropagation();
                            const startOffset = labelOffsets[region.regionId] ?? {
                              xPercent: 0,
                              yPercent: 0,
                            };
                            draggingLabelRef.current = {
                              regionId: region.regionId,
                              startClientX: event.clientX,
                              startClientY: event.clientY,
                              startOffset,
                              moved: false,
                            };
                            setDraggingLabelId(region.regionId);
                          }}
                          onClick={(event) => {
                            if (suppressLabelClickRef.current) return;
                            event.stopPropagation();
                            advanceCellState(region.regionId);
                          }}
                          style={{
                            left: `${(labelX / current.width) * 100 + offset.xPercent}%`,
                            top: `${(labelY / current.height) * 100 + offset.yPercent}%`,
                          }}
                          title="Click to cycle EMPTY/NON_EMPTY. Drag to move."
                        >
                          {regionLabel(layout, region)}
                        </div>
                      );
                    })}
                </div>
              </div>
            </div>
          ) : (
            <div className="annotation-empty">
              <strong>No image selected</strong>
              <p>Upload images or select one from the dataset.</p>
            </div>
          )}
          <div className="canvas-hint">
            <span>{guideText.body}</span>
          </div>
        </section>

        <aside className="panel annotation-sidebar">
          <div className="sidebar-heading">
            <div>
              <p className="step-label">annotation</p>
              <h2>{current?.originalFilename ?? "Image details"}</h2>
            </div>
            <span className={`save-state ${saveState}`}>{saveState}</span>
          </div>

          <div className="annotation-fields">
            <label>
              <span>Dataset split</span>
              <select
                value={split}
                onChange={(event) => {
                  setSplit(event.target.value as DatasetSplit);
                  markDirty();
                }}
              >
                <option value="train">Train</option>
                <option value="valid">Validation</option>
                <option value="test">Test</option>
              </select>
            </label>
            <label>
              <span>Tote layout</span>
              <select value={layout} onChange={(event) => handleLayoutChange(event.target.value as ToteLayout)}>
                <option value="UNKNOWN">Unknown</option>
                <option value="OPEN">Open tote</option>
                <option value="TWO_CELL">Two cell</option>
                <option value="FOUR_CELL">Four cell</option>
              </select>
            </label>
          </div>

          <div className="wizard-steps">
            <section className={`wizard-step ${wizard.toteComplete ? "complete" : ""} ${wizard.layoutSelected && !wizard.toteComplete ? "active" : ""}`}>
              <header>
                <span>2</span>
                <div>
                  <strong>Outline tote</strong>
                  <small>
                    {wizard.toteComplete
                      ? "Complete"
                      : wizard.layoutSelected
                        ? "Draw the outside tote boundary"
                        : "Select a layout first"}
                  </small>
                </div>
              </header>
              <label className="wizard-shape">
                <span>Outline shape</span>
                <select
                  value={shape}
                  onChange={(event) => {
                    setShape(event.target.value as "polygon" | "box");
                    clearDraft();
                  }}
                >
                  <option value="polygon">Polygon</option>
                  <option value="box">Box</option>
                </select>
              </label>
              <button
                className="secondary-action"
                disabled={!wizard.layoutSelected}
                onClick={() => selectTool("tote")}
                type="button"
              >
                Draw tote outline
              </button>
            </section>

            <section className={`wizard-step ${wizard.dividersComplete ? "complete" : ""} ${wizard.toteComplete && !wizard.dividersComplete ? "active" : ""}`}>
              <header>
                <span>3</span>
                <div>
                  <strong>Mark divider</strong>
                  <small>
                    {layout === "OPEN"
                      ? wizard.toteComplete
                        ? "Not needed; cell A uses the tote outline"
                        : "Waiting for tote"
                      : !wizard.toteComplete
                        ? "Waiting for tote outline"
                        : `${wizard.dividers.length} of ${wizard.dividerTarget} marked`}
                  </small>
                </div>
              </header>
              <p>
                {layout === "FOUR_CELL"
                  ? "Mark both dividers. Each line must cross the full tote outline."
                  : layout === "OPEN"
                    ? "Open totes have no divider. Cell A is created automatically."
                    : "Mark the divider with one line crossing the full tote outline."}
              </p>
              <button
                className="secondary-action"
                disabled={
                  !wizard.toteComplete ||
                  wizard.dividerTarget === 0 ||
                  wizard.dividers.length >= wizard.dividerTarget
                }
                onClick={() => selectTool("divider")}
                type="button"
              >
                {wizard.dividerTarget > 1
                  ? `Draw divider ${Math.min(wizard.dividers.length + 1, wizard.dividerTarget)}`
                  : "Draw divider"}
              </button>
            </section>

            <section className={`wizard-step ${wizard.labelsComplete ? "complete" : ""} ${wizard.dividersComplete && !wizard.labelsComplete ? "active" : ""}`}>
              <header>
                <span>4</span>
                <div>
                  <strong>Label cells</strong>
                  <small>
                    {wizard.cellsComplete
                      ? wizard.labelsComplete
                        ? "All cells labeled"
                        : `${wizard.cells.filter((cell) => cell.cellState).length} of ${wizard.cellTarget} labeled`
                      : "Waiting for generated cells"}
                  </small>
                </div>
              </header>
            </section>
          </div>

          <div className="region-heading">
            <p className="step-label">cell labels</p>
            <span>{wizard.cells.length}</span>
          </div>
          <div className="region-list">
            {wizard.cells.map((region) => (
              <div className="region-card" key={region.regionId}>
                <div className="region-card-header">
                  <span className="region-type">{`Cell ${region.cellId}`}</span>
                </div>
                <div className="region-controls">
                  <input readOnly value={region.cellId ?? ""} />
                  <select
                    value={region.cellState ?? ""}
                    onChange={(event) => {
                      const nextState = (event.target.value || null) as CellState | null;
                      setRegions((currentRegions) =>
                        currentRegions.map((candidate) =>
                          candidate.regionId === region.regionId
                            ? { ...candidate, cellState: nextState }
                            : candidate,
                        ),
                      );
                      markDirty();
                    }}
                  >
                    <option value="">Select state</option>
                    <option value="EMPTY">EMPTY</option>
                    <option value="NON_EMPTY">NON_EMPTY</option>
                  </select>
                </div>
              </div>
            ))}
          </div>

          <label className="ready-toggle">
            <input checked={wizard.complete} disabled readOnly type="checkbox" />
            <span>
              <strong>Ready for training</strong>
              <small>Validates and exports this image.</small>
            </span>
          </label>
          <button
            disabled={!current || saving}
            id="save-annotation"
            onClick={() => void handleSave()}
            type="button"
          >
            <span className="button-label">
              {saving ? <span className="button-spinner" aria-hidden="true" /> : null}
              <span>
                {saving
                  ? wizard.complete
                    ? "Saving and exporting..."
                    : "Saving draft..."
                  : wizard.complete
                    ? "Save and export"
                    : "Save draft"}
              </span>
            </span>
            <span>→</span>
          </button>
          <p className="form-error" role="alert">
            {error}
          </p>
        </aside>
      </main>
    </>
  );
}
