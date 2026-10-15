import { NextResponse } from "next/server";
import { profileContext, splitPassages } from "@/lib/profiler";

export const runtime = "nodejs";
export const dynamic = "force-dynamic";

/**
 * POST /api/profile
 * body: { query: string, context?: string, passages?: string[], modelWindow?: number }
 *
 * Returns a structural risk profile. Raw passage text is NEVER echoed back -
 * only short previews - so customer context is not persisted by the endpoint.
 */
export async function POST(req: Request) {
  let body: unknown;
  try {
    body = await req.json();
  } catch {
    return NextResponse.json({ error: "invalid JSON body" }, { status: 400 });
  }
  const b = (body ?? {}) as Record<string, unknown>;

  const query = typeof b.query === "string" ? b.query : "";
  let passages: string[] = [];
  if (Array.isArray(b.passages)) passages = b.passages.map(String);
  else if (typeof b.context === "string") passages = splitPassages(b.context);

  if (!query.trim() || passages.length === 0) {
    return NextResponse.json(
      { error: "both 'query' and ('context' or 'passages') are required" },
      { status: 400 },
    );
  }
  if (passages.length > 200) {
    return NextResponse.json(
      { error: "too many passages (max 200)" },
      { status: 413 },
    );
  }

  const modelWindow =
    typeof b.modelWindow === "number" && b.modelWindow > 0
      ? b.modelWindow
      : undefined;

  const url = new URL(req.url);
  const debug =
    url.searchParams.get("debug") === "1" || b.debug === true;

  const result = profileContext({ query, passages, modelWindow, debug });

  // privacy: return previews, not the raw passages
  const safe = {
    ...result,
    passages: result.passages.map((p) => ({ ...p, text: p.preview })),
    optimized: {
      ...result.optimized,
      passages: result.optimized.passages.map((p) => ({ ...p, text: p.preview })),
    },
  };
  return NextResponse.json(safe, {
    headers: { "cache-control": "no-store" },
  });
}

export async function GET() {
  return NextResponse.json({
    endpoint: "POST /api/profile",
    engineVersion: "profiler-1.0",
    body: {
      query: "string (required)",
      context: "string (optional) - blank-line separated passages",
      passages: "string[] (optional) - use instead of context",
      modelWindow: "number (optional) - default 128000",
    },
  });
}