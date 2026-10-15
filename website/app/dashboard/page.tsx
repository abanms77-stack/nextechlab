import type { Metadata } from "next";
import { Dashboard } from "@/components/Dashboard";

export const metadata: Metadata = {
  title: "Dashboard",
  description: "Analyze contexts, review runs, manage API keys.",
};

export default function DashboardPage() {
  return (
    <section className="container-tight py-12">
      <div className="eyebrow mb-3">dashboard</div>
      <h1 className="text-2xl font-semibold tracking-tight text-ink-900 sm:text-3xl">
        Workspace
      </h1>
      <div className="mt-8">
        <Dashboard />
      </div>
    </section>
  );
}