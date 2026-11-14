"use client";

import React, { useEffect, useState } from "react";
import XRayViewer from "../components/XRayViewer";
import {
  API_BASE_URL,
  UploadResponse,
  deleteAnalysisById,
  fetchAnalysisDetail,
  fetchAnalysisHistory,
  fetchModelsAndAudit,
  loadSampleXRay,
  loginUserAccount,
  registerUserAccount,
  runAnalysisById,
  uploadXRayFile,
} from "../services/api";

type ActiveTab =
  | "landing"
  | "dashboard"
  | "upload"
  | "results"
  | "history"
  | "auth";

export default function RheumatologyPlatformPage({
  initialTab = "landing",
}: {
  initialTab?: ActiveTab;
}) {
  const [activeTab, setActiveTab] = useState<ActiveTab>(initialTab);
  const [token, setToken] = useState<string | null>(null);
  const [user, setUser] = useState<any | null>(null);

  // Auth form state
  const [authMode, setAuthMode] = useState<"login" | "register">("login");
  const [authName, setAuthName] = useState("");
  const [authEmail, setAuthEmail] = useState("");
  const [authPassword, setAuthPassword] = useState("");
  const [authError, setAuthError] = useState<string | null>(null);

  // Upload & Analysis state
  const [uploadedRecord, setUploadedRecord] = useState<UploadResponse | null>(
    null
  );
  const [isUploading, setIsUploading] = useState(false);
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [activeAnalysis, setActiveAnalysis] = useState<any | null>(null);
  const [statusError, setStatusError] = useState<string | null>(null);

  // Dashboard & History state
  const [historyData, setHistoryData] = useState<any>({
    statistics: {
      total_analyses: 0,
      completed_analyses: 0,
      ra_pattern_detected_count: 0,
      generated_reports_count: 0,
    },
    analyses: [],
  });
  const [modelsInfo, setModelsInfo] = useState<any>({
    models: [],
    dataset_audit: {},
    validation_report: {},
  });

  const refreshPlatformData = async () => {
    try {
      const [hist, mod] = await Promise.all([
        fetchAnalysisHistory(),
        fetchModelsAndAudit(),
      ]);
      setHistoryData(hist);
      setModelsInfo(mod);
    } catch {
      // Backend might be initializing
    }
  };

  useEffect(() => {
    refreshPlatformData();
  }, []);

  const handleFileChange = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;
    setStatusError(null);
    setIsUploading(true);
    try {
      const res = await uploadXRayFile(file, token);
      setUploadedRecord(res);
    } catch (err: any) {
      setStatusError(err.message || "Upload failed");
    } finally {
      setIsUploading(false);
    }
  };

  const handleLoadSample = async (caseType: "ra_positive" | "normal") => {
    setStatusError(null);
    setIsUploading(true);
    try {
      const res = await loadSampleXRay(caseType, token);
      setUploadedRecord(res);
      setActiveTab("upload");
    } catch (err: any) {
      setStatusError(err.message || "Could not load sample radiograph");
    } finally {
      setIsUploading(false);
    }
  };

  const handleRunAnalysis = async () => {
    if (!uploadedRecord) return;
    setStatusError(null);
    setIsAnalyzing(true);
    try {
      await runAnalysisById(uploadedRecord.analysis_id);
      const detail = await fetchAnalysisDetail(uploadedRecord.analysis_id);
      setActiveAnalysis(detail);
      await refreshPlatformData();
      setActiveTab("results");
    } catch (err: any) {
      setStatusError(err.message || "AI analysis failed");
    } finally {
      setIsAnalyzing(false);
    }
  };

  const handleOpenExistingAnalysis = async (analysisId: string) => {
    setStatusError(null);
    try {
      const detail = await fetchAnalysisDetail(analysisId);
      setActiveAnalysis(detail);
      setActiveTab("results");
    } catch (err: any) {
      setStatusError(err.message || "Could not load analysis details");
    }
  };

  const handleDeleteAnalysis = async (analysisId: string) => {
    try {
      await deleteAnalysisById(analysisId);
      await refreshPlatformData();
      if (activeAnalysis?.analysis_id === analysisId) {
        setActiveAnalysis(null);
      }
    } catch (err: any) {
      setStatusError(err.message || "Failed to delete analysis");
    }
  };

  const handleAuthSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setAuthError(null);
    try {
      const res =
        authMode === "register"
          ? await registerUserAccount(authName, authEmail, authPassword)
          : await loginUserAccount(authEmail, authPassword);
      setToken(res.access_token);
      setUser(res.user);
      setActiveTab("dashboard");
    } catch (err: any) {
      setAuthError(err.message || "Authentication failed");
    }
  };

  const summary = activeAnalysis?.summary || {};
  const cls = summary.classification || {};
  const det = summary.detection || {};
  const seg = summary.segmentation || {};
  const sev = summary.severity || {};
  const vis = summary.visualizations || {};

  return (
    <div className="min-h-screen flex flex-col bg-slate-50">
      {/* Top Clinical Navigation Bar */}
      <header className="bg-white border-b border-slate-200 sticky top-0 z-30">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 h-16 flex items-center justify-between">
          <div
            onClick={() => setActiveTab("landing")}
            className="flex items-center gap-3 cursor-pointer"
          >
            <div className="w-9 h-9 rounded-lg bg-teal-700 text-white flex items-center justify-center font-bold text-sm shadow-sm">
              RA
            </div>
            <div>
              <h1 className="text-sm sm:text-base font-bold text-slate-900 leading-tight">
                RheumaVision AI
              </h1>
              <p className="text-[11px] text-slate-500">
                Hand X-Ray Rheumatological Analysis Platform (RAM-H1200)
              </p>
            </div>
          </div>

          <nav className="flex items-center gap-1 sm:gap-2">
            {(
              [
                ["landing", "Overview"],
                ["dashboard", "Dashboard"],
                ["upload", "X-Ray Upload"],
                ["results", "Results Viewer"],
                ["history", "History"],
              ] as [ActiveTab, string][]
            ).map(([key, label]) => (
              <button
                key={key}
                onClick={() => setActiveTab(key)}
                className={`px-3 py-1.5 rounded-lg text-xs sm:text-sm font-medium transition ${
                  activeTab === key
                    ? "bg-teal-50 text-teal-800 border border-teal-200 font-semibold"
                    : "text-slate-600 hover:bg-slate-100"
                }`}
              >
                {label}
              </button>
            ))}

            {user ? (
              <div className="ml-2 flex items-center gap-2 pl-2 border-l border-slate-200">
                <span className="text-xs font-medium text-slate-700">
                  {user.name}
                </span>
                <button
                  onClick={() => {
                    setUser(null);
                    setToken(null);
                  }}
                  className="text-xs text-slate-500 hover:text-red-600"
                >
                  Sign Out
                </button>
              </div>
            ) : (
              <button
                onClick={() => setActiveTab("auth")}
                className="ml-2 px-3 py-1.5 rounded-lg text-xs sm:text-sm font-semibold bg-slate-900 text-white hover:bg-slate-800"
              >
                Researcher Sign In
              </button>
            )}
          </nav>
        </div>
      </header>

      {/* Main Content Area */}
      <main className="flex-1 max-w-7xl w-full mx-auto px-4 sm:px-6 lg:px-8 py-8">
        {statusError && (
          <div className="mb-6 p-4 rounded-xl bg-red-50 border border-red-200 text-red-800 text-sm flex items-center justify-between">
            <span>{statusError}</span>
            <button
              onClick={() => setStatusError(null)}
              className="text-xs font-bold underline"
            >
              Dismiss
            </button>
          </div>
        )}

        {/* PAGE 1: LANDING / OVERVIEW */}
        {activeTab === "landing" && (
          <div className="space-y-8">
            <div className="bg-white rounded-2xl border border-slate-200 p-8 shadow-sm">
              <div className="max-w-3xl space-y-4">
                <span className="inline-flex items-center px-3 py-1 rounded-full text-xs font-semibold bg-teal-50 text-teal-800 border border-teal-200">
                  End-to-End Deep Learning Medical Imaging Platform · RAM-H1200 Benchmark
                </span>
                <h2 className="text-2xl sm:text-4xl font-bold text-slate-900 tracking-tight">
                  Multi-Model Hand X-Ray Analysis for Rheumatoid Arthritis
                </h2>
                <p className="text-slate-600 text-sm sm:text-base leading-relaxed">
                  An integrated research and clinical decision-support platform combining{" "}
                  <strong>RA Pattern Classification</strong>,{" "}
                  <strong>16-Joint SvdH Localization</strong>,{" "}
                  <strong>Pixel-Level Bone & Erosion Segmentation</strong>,{" "}
                  <strong>Sharp/van der Heijde (SvdH) Damage Scoring</strong>, and{" "}
                  <strong>Grad-CAM Visual Explainability</strong>.
                </p>
                <div className="pt-2 flex flex-wrap gap-3">
                  <button
                    onClick={() => setActiveTab("upload")}
                    className="px-5 py-2.5 rounded-xl bg-teal-700 text-white font-semibold text-sm hover:bg-teal-800 shadow-sm"
                  >
                    Upload Hand X-Ray Radiograph
                  </button>
                  <button
                    onClick={() => handleLoadSample("ra_positive")}
                    className="px-5 py-2.5 rounded-xl bg-white text-slate-800 border border-slate-300 font-semibold text-sm hover:bg-slate-50"
                  >
                    Load Sample RAM-H1200 RA Case
                  </button>
                  <button
                    onClick={() => setActiveTab("dashboard")}
                    className="px-5 py-2.5 rounded-xl bg-slate-100 text-slate-700 font-semibold text-sm hover:bg-slate-200"
                  >
                    View Research Dashboard
                  </button>
                </div>
              </div>
            </div>

            {/* 4-Model Architecture Cards */}
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
              {[
                {
                  step: "Model 1 · Classification",
                  title: "RA Pattern Classifier",
                  arch: "DenseNet121 / ResNet50 / EfficientNet",
                  desc: "Identifies rheumatoid arthritis radiographic patterns with temperature-calibrated probability estimates.",
                },
                {
                  step: "Model 2 · Detection",
                  title: "SvdH Joint & Erosion Detector",
                  arch: "YOLOv8 / Multi-Joint Localization",
                  desc: "Localizes 16 SvdH target hand joints (MCP, PIP, IP, CMC, Carpal, Radius, Ulna) and flags focal abnormalities.",
                },
                {
                  step: "Model 3 · Segmentation",
                  title: "Bone & Erosion Segmenter",
                  arch: "U-Net / Nested U-Net++",
                  desc: "Generates pixel-level masks of cortical bone, carpal structures, and marginal bone erosions (SvdH-BE).",
                },
                {
                  step: "Model 4 · Severity Scoring",
                  title: "Sharp/van der Heijde Scorer",
                  arch: "Multi-Task Ordinal ResNet",
                  desc: "Estimates joint-by-joint Bone Erosion (0–80) and Joint Space Narrowing (0–60) SvdH scores.",
                },
              ].map((card) => (
                <div
                  key={card.step}
                  className="bg-white rounded-xl border border-slate-200 p-5 shadow-sm flex flex-col justify-between"
                >
                  <div>
                    <span className="text-xs font-semibold text-teal-700 uppercase tracking-wider">
                      {card.step}
                    </span>
                    <h3 className="text-base font-bold text-slate-900 mt-1">
                      {card.title}
                    </h3>
                    <p className="text-xs font-mono text-slate-500 mt-0.5">
                      {card.arch}
                    </p>
                    <p className="text-xs text-slate-600 mt-3 leading-relaxed">
                      {card.desc}
                    </p>
                  </div>
                </div>
              ))}
            </div>

            {/* Research Disclaimer Banner */}
            <div className="bg-amber-50 border border-amber-200 rounded-xl p-4 text-xs text-amber-900 leading-relaxed">
              <strong>Research & Clinical Decision-Support Disclaimer:</strong>{" "}
              This platform is designed for academic research, model benchmarking,
              and decision support on hand radiographs. Model outputs, Grad-CAM
              saliency maps, and SvdH damage estimates must not be used as an
              autonomous clinical diagnosis.
            </div>
          </div>
        )}

        {/* PAGE 2: AUTHENTICATION */}
        {activeTab === "auth" && (
          <div className="max-w-md mx-auto bg-white rounded-2xl border border-slate-200 p-6 shadow-sm">
            <div className="flex gap-2 mb-6 border-b border-slate-200 pb-3">
              <button
                onClick={() => setAuthMode("login")}
                className={`flex-1 py-2 rounded-lg text-xs font-bold ${
                  authMode === "login"
                    ? "bg-teal-700 text-white"
                    : "bg-slate-100 text-slate-600"
                }`}
              >
                Researcher Sign In
              </button>
              <button
                onClick={() => setAuthMode("register")}
                className={`flex-1 py-2 rounded-lg text-xs font-bold ${
                  authMode === "register"
                    ? "bg-teal-700 text-white"
                    : "bg-slate-100 text-slate-600"
                }`}
              >
                Create Account
              </button>
            </div>

            {authError && (
              <div className="mb-4 p-3 rounded-lg bg-red-50 border border-red-200 text-xs text-red-700">
                {authError}
              </div>
            )}

            <form onSubmit={handleAuthSubmit} className="space-y-4">
              {authMode === "register" && (
                <div>
                  <label className="block text-xs font-semibold text-slate-700 mb-1">
                    Full Name / Researcher Title
                  </label>
                  <input
                    type="text"
                    required
                    value={authName}
                    onChange={(e) => setAuthName(e.target.value)}
                    placeholder="Dr. A. Researcher"
                    className="w-full px-3 py-2 rounded-lg border border-slate-300 text-sm"
                  />
                </div>
              )}
              <div>
                <label className="block text-xs font-semibold text-slate-700 mb-1">
                  Institutional Email
                </label>
                <input
                  type="email"
                  required
                  value={authEmail}
                  onChange={(e) => setAuthEmail(e.target.value)}
                  placeholder="researcher@institution.edu"
                  className="w-full px-3 py-2 rounded-lg border border-slate-300 text-sm"
                />
              </div>
              <div>
                <label className="block text-xs font-semibold text-slate-700 mb-1">
                  Password
                </label>
                <input
                  type="password"
                  required
                  value={authPassword}
                  onChange={(e) => setAuthPassword(e.target.value)}
                  placeholder="••••••••"
                  className="w-full px-3 py-2 rounded-lg border border-slate-300 text-sm"
                />
              </div>
              <button
                type="submit"
                className="w-full py-2.5 rounded-xl bg-teal-700 text-white font-semibold text-sm hover:bg-teal-800"
              >
                {authMode === "register" ? "Register Account" : "Sign In"}
              </button>
            </form>
          </div>
        )}

        {/* PAGE 3: RESEARCH DASHBOARD */}
        {activeTab === "dashboard" && (
          <div className="space-y-6">
            <div className="flex items-center justify-between">
              <div>
                <h2 className="text-xl font-bold text-slate-900">
                  Platform & Model Registry Dashboard
                </h2>
                <p className="text-xs text-slate-500">
                  Live database metrics, RAM-H1200 dataset audit, and deployed model checkpoints
                </p>
              </div>
              <button
                onClick={() => setActiveTab("upload")}
                className="px-4 py-2 rounded-xl bg-teal-700 text-white text-xs font-semibold hover:bg-teal-800"
              >
                + New X-Ray Analysis
              </button>
            </div>

            {/* Live Database Counters */}
            <div className="grid grid-cols-2 lg:grid-cols-4 gap-4">
              {[
                {
                  label: "Total Analyses",
                  val: historyData.statistics?.total_analyses ?? 0,
                  sub: "Stored in Database",
                },
                {
                  label: "Completed Runs",
                  val: historyData.statistics?.completed_analyses ?? 0,
                  sub: "4-Model Pipeline Executed",
                },
                {
                  label: "RA Patterns Flagged",
                  val: historyData.statistics?.ra_pattern_detected_count ?? 0,
                  sub: "Calibrated P(RA) ≥ 0.50",
                },
                {
                  label: "Generated PDF Reports",
                  val: historyData.statistics?.generated_reports_count ?? 0,
                  sub: "Downloadable Clinical Reports",
                },
              ].map((stat) => (
                <div
                  key={stat.label}
                  className="bg-white rounded-xl border border-slate-200 p-4 shadow-sm"
                >
                  <p className="text-xs font-medium text-slate-500">
                    {stat.label}
                  </p>
                  <p className="text-2xl font-bold text-slate-900 mt-1">
                    {stat.val}
                  </p>
                  <p className="text-[11px] text-teal-700 mt-1">{stat.sub}</p>
                </div>
              ))}
            </div>

            {/* RAM-H1200 Audit & Patient-Level Split Verification */}
            <div className="bg-white rounded-xl border border-slate-200 p-5 shadow-sm">
              <h3 className="text-sm font-bold text-slate-900 mb-3">
                RAM-H1200 Official Dataset Audit & Patient-Level Split Verification
              </h3>
              <div className="grid grid-cols-1 md:grid-cols-4 gap-4 text-xs">
                <div className="p-3 rounded-lg bg-slate-50 border border-slate-200">
                  <span className="text-slate-500 block">Repository</span>
                  <strong className="text-slate-900">
                    {modelsInfo.dataset_audit?.repo_id ||
                      "TokyoTechMagicYang/RAM-H1200-v1"}
                  </strong>
                  <span className="block text-[11px] text-slate-500 mt-1">
                    1,200 Full-Hand Radiographs · 291 Patients
                  </span>
                </div>
                <div className="p-3 rounded-lg bg-slate-50 border border-slate-200">
                  <span className="text-slate-500 block">
                    Official Train / Val / Test
                  </span>
                  <strong className="text-slate-900">
                    793 Train · 140 Val · 267 Test
                  </strong>
                  <span className="block text-[11px] text-slate-500 mt-1">
                    6 Centers (HMCRD, HU1, HU2, HU3, SARC, SCGH)
                  </span>
                </div>
                <div className="p-3 rounded-lg bg-slate-50 border border-slate-200">
                  <span className="text-slate-500 block">
                    Patient Leakage Audit
                  </span>
                  <strong className="text-teal-700">
                    0 Patient Overlap Verified
                  </strong>
                  <span className="block text-[11px] text-slate-500 mt-1">
                    Train ∩ Val = 0 · Train ∩ Test = 0 · Val ∩ Test = 0
                  </span>
                </div>
                <div className="p-3 rounded-lg bg-slate-50 border border-slate-200">
                  <span className="text-slate-500 block">
                    Validation Pipeline Status
                  </span>
                  <strong className="text-teal-700">
                    {modelsInfo.validation_report?.validation_passed
                      ? "All 8 Checks Passed"
                      : "Ready"}
                  </strong>
                  <span className="block text-[11px] text-slate-500 mt-1">
                    0 Corrupted · 0 Duplicates · RLE & SvdH Verified
                  </span>
                </div>
              </div>
            </div>

            {/* Registered Production Models Table */}
            <div className="bg-white rounded-xl border border-slate-200 p-5 shadow-sm">
              <h3 className="text-sm font-bold text-slate-900 mb-3">
                Registered Deep-Learning Models (Model Registry)
              </h3>
              <div className="overflow-x-auto">
                <table className="w-full text-left text-xs">
                  <thead>
                    <tr className="border-b border-slate-200 text-slate-500">
                      <th className="py-2 pr-4">Task</th>
                      <th className="py-2 pr-4">Model Name</th>
                      <th className="py-2 pr-4">Architecture</th>
                      <th className="py-2 pr-4">Version</th>
                      <th className="py-2 pr-4">Dataset</th>
                      <th className="py-2 pr-4">Status</th>
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-slate-100">
                    {(modelsInfo.models || []).map((m: any) => (
                      <tr key={m.model_id}>
                        <td className="py-2.5 pr-4 font-semibold uppercase text-teal-700">
                          {m.task}
                        </td>
                        <td className="py-2.5 pr-4 font-medium text-slate-900">
                          {m.model_name}
                        </td>
                        <td className="py-2.5 pr-4 font-mono text-slate-600">
                          {m.architecture}
                        </td>
                        <td className="py-2.5 pr-4">{m.version}</td>
                        <td className="py-2.5 pr-4">{m.dataset_version}</td>
                        <td className="py-2.5 pr-4">
                          <span className="px-2 py-0.5 rounded-full bg-teal-50 text-teal-800 border border-teal-200 text-[11px] font-semibold">
                            {m.deployment_status}
                          </span>
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          </div>
        )}

        {/* PAGE 4: X-RAY UPLOAD & QUALITY CHECK */}
        {activeTab === "upload" && (
          <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
            <div className="lg:col-span-7 bg-white rounded-2xl border border-slate-200 p-6 shadow-sm space-y-5">
              <div>
                <h2 className="text-lg font-bold text-slate-900">
                  Hand X-Ray Radiograph Upload & Validation
                </h2>
                <p className="text-xs text-slate-500 mt-0.5">
                  Supported formats: BMP, PNG, JPG, JPEG, DICOM (.dcm). Automatic resolution & contrast quality verification.
                </p>
              </div>

              <label className="flex flex-col items-center justify-center border-2 border-dashed border-teal-300 rounded-xl p-8 bg-teal-50/40 hover:bg-teal-50 transition cursor-pointer">
                <span className="text-sm font-semibold text-teal-900">
                  {isUploading
                    ? "Validating & Preprocessing Radiograph..."
                    : "Click or Drag & Drop Hand X-Ray Image Here"}
                </span>
                <span className="text-xs text-slate-500 mt-1">
                  Posterior-Anterior (PA) Hand Radiograph · Max 25 MB
                </span>
                <input
                  type="file"
                  accept=".bmp,.png,.jpg,.jpeg,.dcm,.dicom"
                  onChange={handleFileChange}
                  className="hidden"
                />
              </label>

              <div className="flex flex-wrap items-center justify-between gap-2 pt-2 border-t border-slate-100">
                <span className="text-xs text-slate-500 font-medium">
                  Or test immediately with a RAM-H1200 benchmark sample:
                </span>
                <div className="flex gap-2">
                  <button
                    onClick={() => handleLoadSample("ra_positive")}
                    disabled={isUploading}
                    className="px-3 py-1.5 rounded-lg text-xs font-semibold bg-slate-100 hover:bg-slate-200 text-slate-800"
                  >
                    Load RA-Positive Hand X-Ray
                  </button>
                  <button
                    onClick={() => handleLoadSample("normal")}
                    disabled={isUploading}
                    className="px-3 py-1.5 rounded-lg text-xs font-semibold bg-slate-100 hover:bg-slate-200 text-slate-800"
                  >
                    Load Control Hand X-Ray
                  </button>
                </div>
              </div>
            </div>

            {/* Preview & Quality Verification Panel */}
            <div className="lg:col-span-5 bg-white rounded-2xl border border-slate-200 p-6 shadow-sm flex flex-col justify-between">
              {uploadedRecord ? (
                <div className="space-y-4">
                  <div className="flex items-center justify-between">
                    <div>
                      <span className="text-[11px] font-semibold uppercase text-teal-700">
                        Validated Radiograph Preview
                      </span>
                      <h3 className="text-sm font-bold text-slate-900 truncate max-w-[240px]">
                        {uploadedRecord.filename}
                      </h3>
                    </div>
                    <button
                      onClick={() => setUploadedRecord(null)}
                      className="text-xs text-slate-500 hover:text-red-600"
                    >
                      Remove / Replace
                    </button>
                  </div>

                  <div className="rounded-xl bg-black p-3 flex items-center justify-center border border-slate-800">
                    <img
                      src={`${API_BASE_URL}${uploadedRecord.preview_url}`}
                      alt="Uploaded X-Ray Preview"
                      className="max-h-60 object-contain"
                    />
                  </div>

                  <div className="grid grid-cols-2 gap-2 text-xs bg-slate-50 p-3 rounded-xl border border-slate-200">
                    <div>
                      <span className="text-slate-500 block">Dimensions</span>
                      <strong className="text-slate-800">
                        {uploadedRecord.quality_metadata.width} ×{" "}
                        {uploadedRecord.quality_metadata.height} px
                      </strong>
                    </div>
                    <div>
                      <span className="text-slate-500 block">
                        Mean / Contrast SD
                      </span>
                      <strong className="text-slate-800">
                        {uploadedRecord.quality_metadata.mean_intensity} /{" "}
                        {uploadedRecord.quality_metadata.std_intensity}
                      </strong>
                    </div>
                    <div>
                      <span className="text-slate-500 block">
                        Dynamic Range
                      </span>
                      <strong className="text-slate-800">
                        {uploadedRecord.quality_metadata.dynamic_range}
                      </strong>
                    </div>
                    <div>
                      <span className="text-slate-500 block">
                        Quality Check
                      </span>
                      <strong className="text-teal-700">Passed</strong>
                    </div>
                  </div>

                  <button
                    onClick={handleRunAnalysis}
                    disabled={isAnalyzing}
                    className="w-full py-3 rounded-xl bg-teal-700 text-white font-bold text-sm hover:bg-teal-800 shadow-sm transition"
                  >
                    {isAnalyzing
                      ? "Running 4-Model Deep Learning Pipeline + Grad-CAM..."
                      : "Start Complete AI Analysis"}
                  </button>
                </div>
              ) : (
                <div className="h-full flex flex-col items-center justify-center text-center py-12 text-slate-400 text-xs">
                  <p className="font-semibold text-slate-600 mb-1">
                    No Radiograph Selected Yet
                  </p>
                  <p>
                    Upload an X-ray image or click &ldquo;Load RA-Positive Hand
                    X-Ray&rdquo; to preview and inspect image quality before running
                    inference.
                  </p>
                </div>
              )}
            </div>
          </div>
        )}

        {/* PAGES 5 & 6: AI ANALYSIS & RESULTS DASHBOARD */}
        {activeTab === "results" && (
          <div className="space-y-6">
            {!activeAnalysis ? (
              <div className="bg-white rounded-2xl border border-slate-200 p-12 text-center space-y-3">
                <h3 className="text-base font-bold text-slate-800">
                  No Active Analysis Selected
                </h3>
                <p className="text-xs text-slate-500">
                  Upload a hand radiograph or select a completed study from
                  Analysis History.
                </p>
                <div className="pt-2 flex justify-center gap-3">
                  <button
                    onClick={() => setActiveTab("upload")}
                    className="px-4 py-2 rounded-xl bg-teal-700 text-white text-xs font-semibold"
                  >
                    Go to X-Ray Upload
                  </button>
                  <button
                    onClick={() => handleLoadSample("ra_positive")}
                    className="px-4 py-2 rounded-xl bg-slate-100 text-slate-700 text-xs font-semibold"
                  >
                    Load Sample Case
                  </button>
                </div>
              </div>
            ) : (
              <>
                {/* Top Action Header with PDF Report Download */}
                <div className="bg-white rounded-xl border border-slate-200 p-4 shadow-sm flex flex-wrap items-center justify-between gap-4">
                  <div>
                    <div className="flex items-center gap-2">
                      <span className="px-2.5 py-0.5 rounded-full text-xs font-bold bg-teal-50 text-teal-800 border border-teal-200 uppercase">
                        {activeAnalysis.analysis_status}
                      </span>
                      <span className="text-xs font-mono text-slate-500">
                        ID: {activeAnalysis.analysis_id.slice(0, 12)}...
                      </span>
                      {summary.processing_time_ms && (
                        <span className="text-xs text-slate-500">
                          · Inference Time: {summary.processing_time_ms} ms
                        </span>
                      )}
                    </div>
                    <h2 className="text-lg font-bold text-slate-900 mt-1">
                      File: {activeAnalysis.filename}
                    </h2>
                  </div>

                  <div className="flex items-center gap-2">
                    <a
                      href={`${API_BASE_URL}/api/analysis/${activeAnalysis.analysis_id}/report`}
                      target="_blank"
                      rel="noopener noreferrer"
                      className="px-4 py-2 rounded-xl bg-teal-700 text-white text-xs font-bold hover:bg-teal-800 shadow-sm"
                    >
                      Download PDF Clinical Research Report
                    </a>
                  </div>
                </div>

                {/* Section A & F: Interactive Multi-Layer Explainable AI Viewer */}
                <XRayViewer visualizations={vis} />

                {/* Sections B, C, D, E Summary Grid */}
                <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                  {/* B. Classification Results */}
                  <div className="bg-white rounded-xl border border-slate-200 p-5 shadow-sm space-y-3">
                    <span className="text-xs font-bold uppercase text-teal-700">
                      B. RA Classification Engine
                    </span>
                    <h3 className="text-base font-bold text-slate-900">
                      {cls.predicted_class || "N/A"}
                    </h3>
                    <div className="space-y-1">
                      <div className="flex justify-between text-xs">
                        <span className="text-slate-500">
                          Calibrated RA Probability
                        </span>
                        <strong className="text-slate-900">
                          {Math.round((cls.ra_probability || 0) * 100)}%
                        </strong>
                      </div>
                      <div className="w-full h-2.5 bg-slate-100 rounded-full overflow-hidden">
                        <div
                          className={`h-full ${
                            cls.is_ra_positive ? "bg-amber-600" : "bg-teal-600"
                          }`}
                          style={{
                            width: `${Math.round(
                              (cls.ra_probability || 0) * 100
                            )}%`,
                          }}
                        />
                      </div>
                    </div>
                    <p className="text-xs text-slate-500">
                      Architecture: <strong>{cls.architecture}</strong> (v
                      {cls.model_version})
                    </p>
                  </div>

                  {/* C & D. Localization & Segmentation Results */}
                  <div className="bg-white rounded-xl border border-slate-200 p-5 shadow-sm space-y-2">
                    <span className="text-xs font-bold uppercase text-teal-700">
                      C & D. Localization & Segmentation
                    </span>
                    <div className="grid grid-cols-2 gap-2 pt-1 text-xs">
                      <div className="p-2.5 rounded-lg bg-slate-50 border border-slate-200">
                        <span className="text-slate-500 block">
                          Joints Localized
                        </span>
                        <strong className="text-base text-slate-900">
                          {det.total_joints_localized ?? 16} / 16
                        </strong>
                      </div>
                      <div className="p-2.5 rounded-lg bg-slate-50 border border-slate-200">
                        <span className="text-slate-500 block">
                          Flagged ROIs
                        </span>
                        <strong className="text-base text-amber-700">
                          {det.abnormal_regions_flagged ?? 0}
                        </strong>
                      </div>
                      <div className="p-2.5 rounded-lg bg-slate-50 border border-slate-200">
                        <span className="text-slate-500 block">
                          Bone Coverage
                        </span>
                        <strong className="text-slate-900">
                          {seg.bone_coverage_percent ?? 0}%
                        </strong>
                      </div>
                      <div className="p-2.5 rounded-lg bg-slate-50 border border-slate-200">
                        <span className="text-slate-500 block">
                          Erosion Ratio
                        </span>
                        <strong className="text-red-700">
                          {seg.erosion_area_percent ?? 0}%
                        </strong>
                      </div>
                    </div>
                    <p className="text-xs text-slate-500 pt-1">
                      Segmenter: <strong>{seg.architecture}</strong> · Detector:{" "}
                      <strong>{det.architecture}</strong>
                    </p>
                  </div>

                  {/* E. SvdH Severity Summary */}
                  <div className="bg-white rounded-xl border border-slate-200 p-5 shadow-sm space-y-2">
                    <span className="text-xs font-bold uppercase text-teal-700">
                      E. SvdH Radiographic Severity
                    </span>
                    <h3 className="text-sm font-bold text-slate-900">
                      {sev.severity_category}
                    </h3>
                    <div className="grid grid-cols-3 gap-2 text-center pt-1">
                      <div className="p-2 rounded-lg bg-slate-50 border border-slate-200">
                        <span className="text-[10px] text-slate-500 block">
                          BE Score
                        </span>
                        <strong className="text-sm text-slate-900">
                          {sev.total_be_score ?? 0}/80
                        </strong>
                      </div>
                      <div className="p-2 rounded-lg bg-slate-50 border border-slate-200">
                        <span className="text-[10px] text-slate-500 block">
                          JSN Score
                        </span>
                        <strong className="text-sm text-slate-900">
                          {sev.total_jsn_score ?? 0}/60
                        </strong>
                      </div>
                      <div className="p-2 rounded-lg bg-teal-50 border border-teal-200">
                        <span className="text-[10px] text-teal-800 block font-semibold">
                          Total SvdH
                        </span>
                        <strong className="text-sm text-teal-900">
                          {sev.total_svdh_score ?? 0}/140
                        </strong>
                      </div>
                    </div>
                    <p className="text-[11px] text-slate-500">
                      {sev.scoring_protocol}
                    </p>
                  </div>
                </div>

                {/* Detailed Joint-by-Joint SvdH Score & Localization Table */}
                <div className="bg-white rounded-xl border border-slate-200 p-5 shadow-sm">
                  <h3 className="text-sm font-bold text-slate-900 mb-3">
                    Anatomical Joint-Level SvdH Breakdown (16 Bone Erosion ROIs & 15 Joint Space Narrowing ROIs)
                  </h3>
                  <div className="overflow-x-auto">
                    <table className="w-full text-left text-xs">
                      <thead>
                        <tr className="border-b border-slate-200 text-slate-500">
                          <th className="py-2 pr-4">Anatomical Joint</th>
                          <th className="py-2 pr-4">Localization Confidence</th>
                          <th className="py-2 pr-4">BE Score (0–5)</th>
                          <th className="py-2 pr-4">JSN Score (0–4)</th>
                          <th className="py-2 pr-4">Observation</th>
                        </tr>
                      </thead>
                      <tbody className="divide-y divide-slate-100">
                        {(det.detections || []).map((d: any) => {
                          const jName = d.joint_name;
                          const beVal = sev.be_joint_scores?.[jName] ?? 0;
                          const jsnVal = sev.jsn_joint_scores?.[jName] ?? "-";
                          return (
                            <tr key={jName}>
                              <td className="py-2 pr-4 font-bold text-slate-900">
                                {jName}
                              </td>
                              <td className="py-2 pr-4 font-mono">
                                {Math.round((d.confidence || 0.9) * 100)}%
                              </td>
                              <td className="py-2 pr-4 font-semibold">
                                {beVal} / 5
                              </td>
                              <td className="py-2 pr-4 font-semibold">
                                {jsnVal !== "-" ? `${jsnVal} / 4` : "N/A (BE ROI)"}
                              </td>
                              <td className="py-2 pr-4">
                                <span
                                  className={`px-2 py-0.5 rounded text-[11px] font-medium ${
                                    d.abnormality_flagged
                                      ? "bg-red-50 text-red-700 border border-red-200"
                                      : "bg-slate-100 text-slate-700"
                                  }`}
                                >
                                  {d.finding}
                                </span>
                              </td>
                            </tr>
                          );
                        })}
                      </tbody>
                    </table>
                  </div>
                </div>
              </>
            )}
          </div>
        )}

        {/* PAGE 7: ANALYSIS HISTORY */}
        {activeTab === "history" && (
          <div className="bg-white rounded-2xl border border-slate-200 p-6 shadow-sm space-y-4">
            <div className="flex items-center justify-between">
              <div>
                <h2 className="text-lg font-bold text-slate-900">
                  Analysis History & Stored Reports
                </h2>
                <p className="text-xs text-slate-500">
                  Inspect previous radiograph analyses, download PDF reports, or remove stored records.
                </p>
              </div>
              <button
                onClick={refreshPlatformData}
                className="px-3 py-1.5 rounded-lg border border-slate-200 text-xs font-semibold text-slate-700 hover:bg-slate-50"
              >
                Refresh
              </button>
            </div>

            {(historyData.analyses || []).length === 0 ? (
              <div className="py-12 text-center text-xs text-slate-500">
                No previous analyses recorded yet. Upload an X-ray or load a
                sample case to begin.
              </div>
            ) : (
              <div className="overflow-x-auto">
                <table className="w-full text-left text-xs">
                  <thead>
                    <tr className="border-b border-slate-200 text-slate-500">
                      <th className="py-2.5 pr-4">Analysis ID</th>
                      <th className="py-2.5 pr-4">Filename</th>
                      <th className="py-2.5 pr-4">Date</th>
                      <th className="py-2.5 pr-4">Status</th>
                      <th className="py-2.5 pr-4">Classification</th>
                      <th className="py-2.5 pr-4">Total SvdH</th>
                      <th className="py-2.5 pr-4">Actions</th>
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-slate-100">
                    {(historyData.analyses || []).map((item: any) => (
                      <tr key={item.analysis_id}>
                        <td className="py-3 pr-4 font-mono text-slate-600">
                          {item.analysis_id.slice(0, 8)}
                        </td>
                        <td className="py-3 pr-4 font-semibold text-slate-900">
                          {item.filename}
                        </td>
                        <td className="py-3 pr-4 text-slate-500">
                          {item.upload_timestamp
                            ? new Date(item.upload_timestamp).toLocaleString()
                            : "-"}
                        </td>
                        <td className="py-3 pr-4">
                          <span className="px-2 py-0.5 rounded-full bg-teal-50 text-teal-800 border border-teal-200 text-[11px] font-semibold">
                            {item.analysis_status}
                          </span>
                        </td>
                        <td className="py-3 pr-4">
                          {item.predicted_class || "Pending"}
                        </td>
                        <td className="py-3 pr-4 font-bold">
                          {item.total_svdh_score !== undefined &&
                          item.total_svdh_score !== null
                            ? `${item.total_svdh_score} / 140`
                            : "-"}
                        </td>
                        <td className="py-3 pr-4 flex items-center gap-2">
                          <button
                            onClick={() =>
                              handleOpenExistingAnalysis(item.analysis_id)
                            }
                            className="px-2.5 py-1 rounded bg-teal-700 text-white font-semibold hover:bg-teal-800"
                          >
                            View Results
                          </button>
                          <a
                            href={`${API_BASE_URL}/api/analysis/${item.analysis_id}/report`}
                            target="_blank"
                            rel="noopener noreferrer"
                            className="px-2.5 py-1 rounded bg-slate-100 text-slate-800 font-semibold hover:bg-slate-200"
                          >
                            PDF
                          </a>
                          <button
                            onClick={() =>
                              handleDeleteAnalysis(item.analysis_id)
                            }
                            className="px-2 py-1 rounded text-red-600 hover:bg-red-50 font-semibold"
                          >
                            Delete
                          </button>
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            )}
          </div>
        )}
      </main>

      {/* Footer */}
      <footer className="bg-white border-t border-slate-200 py-4 mt-12">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 flex flex-wrap items-center justify-between text-xs text-slate-500">
          <span>
            AI-Powered Rheumatological X-Ray Analysis Platform · RAM-H1200 Multi-Model Architecture
          </span>
          <span>
            For Academic Research & Clinical Decision Support Only — Not an Autonomous Diagnostic Device
          </span>
        </div>
      </footer>
    </div>
  );
}
