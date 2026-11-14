"use client";

import React, { useState } from "react";
import { API_BASE_URL } from "../services/api";

interface XRayViewerProps {
  visualizations: {
    original?: string;
    gradcam?: string;
    detection?: string;
    segmentation?: string;
    combined?: string;
  };
}

type LayerKey =
  | "original"
  | "gradcam"
  | "detection"
  | "segmentation"
  | "combined";

const LAYER_LABELS: { key: LayerKey; title: string; subtitle: string }[] = [
  {
    key: "original",
    title: "Original X-Ray",
    subtitle: "CLAHE Enhanced Radiograph",
  },
  {
    key: "gradcam",
    title: "Classification Grad-CAM",
    subtitle: "Regions Influencing RA Prediction",
  },
  {
    key: "detection",
    title: "Joint Localization",
    subtitle: "16 SvdH Target Joints & Flagged ROIs",
  },
  {
    key: "segmentation",
    title: "Bone & Erosion Mask",
    subtitle: "Cortical Bone, Carpal & Erosion Map",
  },
  {
    key: "combined",
    title: "Combined Multi-Modal",
    subtitle: "Unified Clinical Overlay",
  },
];

export default function XRayViewer({ visualizations }: XRayViewerProps) {
  const [activeLayer, setActiveLayer] = useState<LayerKey>("combined");
  const [zoom, setZoom] = useState<number>(1);
  const [compareSideBySide, setCompareSideBySide] = useState<boolean>(true);

  const resolveUrl = (path?: string) => {
    if (!path) return "";
    return path.startsWith("http") ? path : `${API_BASE_URL}${path}`;
  };

  const activeUrl =
    resolveUrl(visualizations[activeLayer]) ||
    resolveUrl(visualizations.original);
  const originalUrl = resolveUrl(visualizations.original);

  return (
    <div className="bg-white border border-slate-200 rounded-xl shadow-sm overflow-hidden">
      {/* Layer Switcher Toolbar */}
      <div className="p-4 border-b border-slate-200 bg-slate-50 flex flex-wrap items-center justify-between gap-3">
        <div className="flex flex-wrap gap-1.5">
          {LAYER_LABELS.map((item) => {
            const isSelected = activeLayer === item.key;
            return (
              <button
                key={item.key}
                onClick={() => setActiveLayer(item.key)}
                className={`px-3 py-1.5 rounded-lg text-xs font-semibold transition-all ${
                  isSelected
                    ? "bg-teal-700 text-white shadow-sm"
                    : "bg-white text-slate-700 border border-slate-200 hover:bg-slate-100"
                }`}
              >
                {item.title}
              </button>
            );
          })}
        </div>

        {/* Zoom & Side-by-Side Comparison Controls */}
        <div className="flex items-center gap-2">
          <button
            onClick={() => setCompareSideBySide(!compareSideBySide)}
            className={`px-2.5 py-1 rounded text-xs font-medium border ${
              compareSideBySide
                ? "bg-sky-50 border-sky-300 text-sky-800"
                : "bg-white border-slate-200 text-slate-600"
            }`}
          >
            {compareSideBySide ? "Side-by-Side: ON" : "Single View"}
          </button>
          <div className="flex items-center border border-slate-200 rounded bg-white">
            <button
              onClick={() => setZoom((z) => Math.max(1, +(z - 0.25).toFixed(2)))}
              className="px-2 py-1 text-xs font-bold text-slate-700 hover:bg-slate-100"
              title="Zoom Out"
            >
              -
            </button>
            <span className="px-2 text-xs font-mono text-slate-700">
              {Math.round(zoom * 100)}%
            </span>
            <button
              onClick={() => setZoom((z) => Math.min(3, +(z + 0.25).toFixed(2)))}
              className="px-2 py-1 text-xs font-bold text-slate-700 hover:bg-slate-100"
              title="Zoom In"
            >
              +
            </button>
          </div>
        </div>
      </div>

      {/* Radiograph Viewport */}
      <div
        className={`grid ${
          compareSideBySide && activeLayer !== "original"
            ? "grid-cols-1 md:grid-cols-2"
            : "grid-cols-1"
        } gap-4 p-4 bg-slate-950`}
      >
        {compareSideBySide && activeLayer !== "original" && (
          <div className="flex flex-col items-center overflow-auto rounded-lg border border-slate-800 bg-black p-2">
            <span className="text-xs font-medium text-slate-300 mb-2">
              Reference: Original Hand Radiograph
            </span>
            <div className="overflow-auto max-h-[440px] flex items-center justify-center">
              {originalUrl && (
                <img
                  src={originalUrl}
                  alt="Original Hand X-Ray"
                  style={{
                    transform: `scale(${zoom})`,
                    transformOrigin: "center center",
                  }}
                  className="max-h-[400px] w-auto object-contain transition-transform duration-150"
                />
              )}
            </div>
          </div>
        )}

        <div className="flex flex-col items-center overflow-auto rounded-lg border border-slate-800 bg-black p-2">
          <span className="text-xs font-medium text-teal-300 mb-2">
            Active Layer:{" "}
            {LAYER_LABELS.find((l) => l.key === activeLayer)?.title} (
            {LAYER_LABELS.find((l) => l.key === activeLayer)?.subtitle})
          </span>
          <div className="overflow-auto max-h-[440px] flex items-center justify-center">
            {activeUrl ? (
              <img
                src={activeUrl}
                alt={`X-Ray ${activeLayer} visualization`}
                style={{
                  transform: `scale(${zoom})`,
                  transformOrigin: "center center",
                }}
                className="max-h-[400px] w-auto object-contain transition-transform duration-150"
              />
            ) : (
              <div className="text-slate-400 text-sm py-20">
                No visualization available
              </div>
            )}
          </div>
        </div>
      </div>

      {/* Color Legend & Explainability Note */}
      <div className="px-4 py-3 bg-slate-50 border-t border-slate-200 flex flex-wrap items-center justify-between gap-2 text-xs text-slate-600">
        <div className="flex items-center gap-4">
          <span className="flex items-center gap-1.5">
            <span className="w-3 h-3 rounded-sm inline-block bg-teal-500"></span>
            Cortical Bone / Localized Joint
          </span>
          <span className="flex items-center gap-1.5">
            <span className="w-3 h-3 rounded-sm inline-block bg-blue-600"></span>
            Carpal / Joint Space
          </span>
          <span className="flex items-center gap-1.5">
            <span className="w-3 h-3 rounded-sm inline-block bg-red-600"></span>
            Suspected Erosion (BE) / High Saliency
          </span>
        </div>
        <span className="italic text-slate-500">
          Note: Visual explainability overlays indicate model attention, not confirmed pathology.
        </span>
      </div>
    </div>
  );
}
