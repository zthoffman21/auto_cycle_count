import { InspectionPage } from "./InspectionPage";
import { TrainingPage } from "./TrainingPage";

export function App() {
  const pathname = window.location.pathname;
  return pathname === "/train" ? <TrainingPage /> : <InspectionPage />;
}
