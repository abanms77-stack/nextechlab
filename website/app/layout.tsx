import type { Metadata } from "next";
import "./globals.css";
import { Nav } from "@/components/Nav";
import { Footer } from "@/components/Footer";

export const metadata: Metadata = {
  title: {
    default: "ContextLab — Make AI context reliable",
    template: "%s · ContextLab",
  },
  description:
    "ContextLab detects competing information inside your LLM context, estimates retrieval risk, and optimizes context before inference.",
  metadataBase: new URL("https://contextlab.example"),
  openGraph: {
    title: "ContextLab — Make AI context reliable",
    description:
      "Detect competing information, predict retrieval risk, optimize context before inference.",
    type: "website",
  },
  robots: { index: true, follow: true },
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en">
      <body className="min-h-screen bg-white">
        <Nav />
        <main>{children}</main>
        <Footer />
      </body>
    </html>
  );
}