import React from "react";
import { useHashRoute, useTheme } from "./hooks.js";
import Gallery from "./components/Gallery.jsx";
import Viewer from "./components/Viewer.jsx";

export default function App() {
  const route = useHashRoute();
  const theme = useTheme();
  if (route.name === "map") return <Viewer mapId={route.mapId} theme={theme} />;
  return <Gallery theme={theme} query={route.query || {}} />;
}
