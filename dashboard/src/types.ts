export type ToteLayout = "UNKNOWN" | "OPEN" | "TWO_CELL" | "FOUR_CELL";
export type DatasetSplit = "train" | "valid" | "test";
export type RegionClass = "tote" | "divider" | "cell";
export type CellState = "EMPTY" | "NON_EMPTY" | "UNCERTAIN" | "BAD_CAPTURE";
export type InspectionResult = "PASS" | "REVIEW" | "FAIL" | "ERROR";

export type Point = [number, number];

export interface TrainingRegion {
  regionId: string;
  regionClass: RegionClass;
  polygon: Point[];
  cellId: string | null;
  cellState: CellState | null;
}

export interface TrainingImage {
  imageId: string;
  originalFilename: string;
  imageUri: string;
  width: number;
  height: number;
  split: DatasetSplit;
  layout: ToteLayout;
  regions: TrainingRegion[];
  ready: boolean;
  createdAt: string;
  updatedAt: string;
}

export interface TrainingStatus {
  totalImages: number;
  readyImages: number;
  draftImages: number;
  exportedCellCrops: number;
  exportDirectory: string;
  exportInProgress: boolean;
  exportPending: boolean;
  exportError: string | null;
}

export interface InspectionCell {
  cellId: string;
  result: CellState;
  polygon: Point[] | null;
  emptyProbability: number;
  nonEmptyProbability: number;
  uncertainProbability: number;
}

export interface DetectedCell {
  cellId: string;
  polygon: Point[];
}

export interface InspectionResponse {
  inspectionId: string;
  toteId: string;
  imageUri: string;
  result: InspectionResult;
  reasonCode: string;
  observedLayout: ToteLayout | null;
  layoutConfidence: number | null;
  coordinateSpace: "NORMALIZED_100" | string;
  totePolygon: Point[] | null;
  cells: InspectionCell[];
  methodResults: Record<string, InspectionCell[]> | null;
  detectedCells: DetectedCell[];
  geometry: { valid: boolean; issues: string[] } | null;
  modelVersions: Record<string, string>;
  decisionVersion: string;
}
