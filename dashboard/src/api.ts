import type {
  InspectionResponse,
  TrainingDraftPredictionResponse,
  TrainingImage,
  TrainingStatus,
} from "./types";

async function parseJson<T>(response: Response): Promise<T> {
  const payload = (await response.json()) as T & { detail?: unknown };
  if (!response.ok) {
    throw new Error(readError(payload));
  }
  return payload;
}

export function readError(payload: { detail?: unknown }): string {
  if (typeof payload.detail === "string") return payload.detail;
  if (Array.isArray(payload.detail)) {
    return payload.detail
      .map((item) => (item && typeof item === "object" && "msg" in item ? String(item.msg) : "Request failed"))
      .join(", ");
  }
  return "Request failed";
}

export async function inspectImage(formData: FormData): Promise<InspectionResponse> {
  const response = await fetch("/dashboard/inspect", { method: "POST", body: formData });
  return parseJson<InspectionResponse>(response);
}

export async function listTrainingImages(): Promise<TrainingImage[]> {
  const response = await fetch("/training/images");
  return parseJson<TrainingImage[]>(response);
}

export async function uploadTrainingImages(formData: FormData): Promise<TrainingImage[]> {
  const response = await fetch("/training/images", { method: "POST", body: formData });
  return parseJson<TrainingImage[]>(response);
}

export async function getTrainingImage(imageId: string): Promise<TrainingImage> {
  const response = await fetch(`/training/images/${imageId}`);
  return parseJson<TrainingImage>(response);
}

export async function updateTrainingImage(
  imageId: string,
  payload: Pick<TrainingImage, "split" | "layout" | "regions" | "ready">,
): Promise<TrainingImage> {
  const response = await fetch(`/training/images/${imageId}`, {
    method: "PUT",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
  return parseJson<TrainingImage>(response);
}

export async function deleteTrainingImage(imageId: string): Promise<void> {
  const response = await fetch(`/training/images/${imageId}`, { method: "DELETE" });
  if (!response.ok) {
    throw new Error("Image could not be deleted.");
  }
}

export async function deleteAllTrainingImages(): Promise<void> {
  const response = await fetch("/training/images", { method: "DELETE" });
  if (!response.ok) {
    throw new Error("Training images could not be deleted.");
  }
}

export async function getTrainingStatus(): Promise<TrainingStatus> {
  const response = await fetch("/training/status");
  return parseJson<TrainingStatus>(response);
}

export async function predictDraftTrainingImages(
  imageIds: string[],
): Promise<TrainingDraftPredictionResponse> {
  const response = await fetch("/training/images/predict-drafts", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ imageIds, overwrite: false }),
  });
  return parseJson<TrainingDraftPredictionResponse>(response);
}
