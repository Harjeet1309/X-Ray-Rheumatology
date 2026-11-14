import React from "react";
import "../styles/globals.css";

export const metadata = {
  title: "AI-Powered Rheumatological X-Ray Analysis Platform | RAM-H1200",
  description:
    "End-to-End Deep Learning Platform for Hand X-Ray Rheumatoid Arthritis Classification, SvdH Joint Localization, Bone & Erosion Segmentation, and Severity Assessment.",
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en">
      <body className="min-h-screen bg-slate-50 text-slate-900 antialiased">
        {children}
      </body>
    </html>
  );
}
