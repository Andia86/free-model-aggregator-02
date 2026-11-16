// Reading the trial brief.
//
// The brief is one text file (MAKEFUN_PROMPT_FILE) that carries three JSON documents, found by
// their `schema` field: the provider directory, the sealed quality table and the workload
// manifest. This module pulls those documents out of the prose, builds the catalog the entry
// registers from them, and answers the routing questions the planner asks. Everything here is a
// pure function of the brief's text: the same brief gives the same catalog, byte for byte, on
// every run.

export type Wire = 'openai' | 'anthropic' | 'gemini';
export type ClientApi = 'openai' | 'anthropic';

/** One provider from the directory document. `key` is the trial's key for it (may be ''). */
export interface ProviderDoc {
  id: string;
  name: string;
  wire: Wire;
  auth: string;
  path: string;
  key_env: string;
  key: string;
}

/** One row of the quality table: a model, its rating and what it can carry. */
export interface RatingRow {
  provider: string;
  model: string;
  rating_milli: number;
  context_length: number;
  tools: boolean;
  vision: boolean;
  json: boolean;
  free: boolean;
}

/** What one workload request needs from the model that answers it. */
export interface Requires {
  tools?: boolean;
  vision?: boolean;
  json?: boolean;
  min_context?: number;
}

/** One workload request from the manifest, in the order the client sends them. */
export interface ManifestRow {
  id: string;
  api: ClientApi;
  stream: boolean;
  weight: number;
  requires: Requires;
}

export interface Brief {
  providers: ProviderDoc[];
  ratings: RatingRow[];
  manifest: ManifestRow[];
}

export type Json = Record<string, unknown>;

export function isRecord(v: unknown): v is Json {
  return v !== null && typeof v === 'object' && !Array.isArray(v);
}

// The brief's three documents carry schema names of the form
// '<competition>/providers/v1' (and ratings, workload likewise). The competition
// slug differs between runs of this contest family, so match on the suffix:
// the document kind and version are what this parser actually depends on.
const PROVIDERS_SCHEMA_SUFFIX = '/providers/v1';
const RATINGS_SCHEMA_SUFFIX = '/ratings/v1';
const WORKLOAD_SCHEMA_SUFFIX = '/workload/v1';

function schemaEndsWith(doc: Json, suffix: string): boolean {
  return typeof doc.schema === 'string' && doc.schema.endsWith(suffix);
}

/**
 * Every balanced `{...}` span in the text that parses as one JSON object, in order of appearance.
 * Braces inside strings do not count; a span that does not parse is skipped, so prose, code and
 * the documents' own nested objects are all safe to walk over.
 */
function jsonObjectsIn(text: string): Json[] {
  const out: Json[] = [];
  let i = 0;
  while (i < text.length) {
    if (text[i] !== '{') {
      i += 1;
      continue;
    }
    let depth = 0;
    let inString = false;
    let escaped = false;
    let j = i;
    for (; j < text.length; j += 1) {
      const c = text[j];
      if (inString) {
        if (escaped) escaped = false;
        else if (c === '\\') escaped = true;
        else if (c === '"') inString = false;
        continue;
      }
      if (c === '"') inString = true;
      else if (c === '{') depth += 1;
      else if (c === '}') {
        depth -= 1;
        if (depth === 0) break;
      }
    }
    if (depth !== 0) break;
    try {
      const value: unknown = JSON.parse(text.slice(i, j + 1));
      if (isRecord(value)) out.push(value);
    } catch {
      // Not a JSON document: prose that happens to brace.
    }
    i = j + 1;
  }
  return out;
}

function stringOf(v: unknown): string {
  return typeof v === 'string' ? v : '';
}

function wireOf(v: unknown): Wire {
  return v === 'anthropic' || v === 'gemini' ? v : 'openai';
}

function requiresOf(v: unknown): Requires {
  if (!isRecord(v)) return {};
  const out: Requires = {};
  if (v.tools === true) out.tools = true;
  if (v.vision === true) out.vision = true;
  if (v.json === true) out.json = true;
  if (typeof v.min_context === 'number') out.min_context = v.min_context;
  return out;
}

/** The three documents of the brief, parsed out of the prompt file's text. */
export function parseBrief(text: string): Brief {
  let providersDoc: Json | null = null;
  let ratingsDoc: Json | null = null;
  let workloadDoc: Json | null = null;
  for (const doc of jsonObjectsIn(text)) {
    if (schemaEndsWith(doc, PROVIDERS_SCHEMA_SUFFIX) && providersDoc === null) providersDoc = doc;
    else if (schemaEndsWith(doc, RATINGS_SCHEMA_SUFFIX) && ratingsDoc === null) ratingsDoc = doc;
    else if (schemaEndsWith(doc, WORKLOAD_SCHEMA_SUFFIX) && workloadDoc === null) workloadDoc = doc;
  }

  const providers: ProviderDoc[] = [];
  if (providersDoc !== null && Array.isArray(providersDoc.providers)) {
    for (const raw of providersDoc.providers) {
      if (!isRecord(raw) || typeof raw.id !== 'string') continue;
      providers.push({
        id: raw.id,
        name: stringOf(raw.name),
        wire: wireOf(raw.wire),
        auth: stringOf(raw.auth),
        path: stringOf(raw.path),
        key_env: stringOf(raw.key_env),
        key: stringOf(raw.key),
      });
    }
  }

  const ratings: RatingRow[] = [];
  if (ratingsDoc !== null && Array.isArray(ratingsDoc.ratings)) {
    for (const raw of ratingsDoc.ratings) {
      if (!isRecord(raw) || typeof raw.provider !== 'string' || typeof raw.model !== 'string') continue;
      ratings.push({
        provider: raw.provider,
        model: raw.model,
        rating_milli: typeof raw.rating_milli === 'number' ? raw.rating_milli : 0,
        context_length: typeof raw.context_length === 'number' ? raw.context_length : 0,
        tools: raw.tools === true,
        vision: raw.vision === true,
        json: raw.json === true,
        free: raw.free === true,
      });
    }
  }

  const manifest: ManifestRow[] = [];
  if (workloadDoc !== null && Array.isArray(workloadDoc.manifest)) {
    for (const raw of workloadDoc.manifest) {
      if (!isRecord(raw) || typeof raw.id !== 'string') continue;
      manifest.push({
        id: raw.id,
        api: raw.api === 'anthropic' ? 'anthropic' : 'openai',
        stream: raw.stream === true,
        weight: typeof raw.weight === 'number' ? raw.weight : 1,
        requires: requiresOf(raw.requires),
      });
    }
  }

  return { providers, ratings, manifest };
}

/** A model row meets a request's requirements, on the quality table's word. Unknown never meets. */
export function meetsRequirements(row: RatingRow, requires: Requires): boolean {
  if (requires.tools === true && !row.tools) return false;
  if (requires.vision === true && !row.vision) return false;
  if (requires.json === true && !row.json) return false;
  if (requires.min_context !== undefined && row.context_length < requires.min_context) return false;
  return true;
}

/**
 * The catalog document: one entry per provider in directory order, its models sorted by id so
 * the document is stable. Built from the brief alone — the directory gives each provider's path
 * and key variable, the quality table gives each model's free flag and context length — so the
 * registered document and `--list`'s output are the same object by construction.
 */
export function buildCatalog(brief: Brief): Json {
  const providers = brief.providers.map((p) => {
    const models = brief.ratings
      .filter((r) => r.provider === p.id)
      .map((r) => ({ id: r.model, free: r.free, context_length: r.context_length }))
      .sort((a, b) => (a.id < b.id ? -1 : a.id > b.id ? 1 : 0));
    return { id: p.id, base_url: p.path, key_env: p.key_env, wire: p.wire, models };
  });
  return { providers };
}

/** The free model ids of a built catalog, in catalog order: what GET /v1/models must list. */
export function freeModelIds(catalog: Json): string[] {
  const ids: string[] = [];
  const providers = Array.isArray(catalog.providers) ? catalog.providers : [];
  for (const raw of providers) {
    if (!isRecord(raw) || !Array.isArray(raw.models)) continue;
    for (const model of raw.models) {
      if (isRecord(model) && model.free === true && typeof model.id === 'string') ids.push(model.id);
    }
  }
  return ids;
}
