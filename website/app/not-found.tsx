import Link from "next/link";

export default function NotFound() {
  return (
    <section className="container-tight grid min-h-[60vh] place-items-center py-20 text-center">
      <div>
        <div className="mono text-[12px] uppercase tracking-[0.2em] text-ink-400">
          404
        </div>
        <h1 className="mt-3 text-2xl font-semibold text-ink-900">
          This page doesn&apos;t exist.
        </h1>
        <p className="mt-2 text-[14px] text-ink-500">
          The link may be old or mistyped.
        </p>
        <div className="mt-6 flex justify-center gap-3">
          <Link href="/" className="btn-primary">Back home</Link>
          <Link href="/playground" className="btn-ghost">Open playground</Link>
        </div>
      </div>
    </section>
  );
}