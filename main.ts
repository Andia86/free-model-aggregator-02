// The free-model aggregator: one server, two client APIs, every free model
// the trial brief names.
//
// The server serves the OpenAI Chat Completions API and the Anthropic
// Messages API, in JSON and in server-sent events, over the providers the
// brief lists. For each request it plans from the brief's sealed quality
// table: the candidates are the free models that meet the request's
// requirements, best rating first; a provider that refuses hands the
// request to the next candidate. At the trial's start it registers its
// port and catalog with the env, and it exits 0 when the env's /work says
// the workload is done.
//
//   node main.ts --list   print the catalog document and exit
//   node main.ts          serve a trial
//
// Everything the server chooses derives from the brief, the requests and
// the providers' answers — never from the wall clock or from randomness —
// so a replayed trial makes the same choices and leaves the same export.

import { createServer } from 'node:http';
import type { IncomingMessage, ServerResponse } from 'node:http';
import { readFileSync } from 'node:fs';
import {
  buildCatalog,
  freeModelIds,
  isRecord,
  meetsRequirements,
  parseBrief,
} from './lib/brief.ts';
import type { Brief, ClientApi, Json, ProviderDoc, RatingRow, Requires } from './lib/brief.ts';
import {
  anthropicError,
  buildProviderRequest,
  openAIError,
  parseAnthropicRequest,
  parseOpenAIRequest,
  parseProviderAnswer,
  providerHeaders,
  providerPath,
  renderAnthropicJson,
  renderAnthropicSse,
  renderOpenAIJson,
  renderOpenAISse,
} from './lib/wire.ts';
import type { NormalizedAnswer, NormalizedRequest } from './lib/wire.ts';

const MAX_FAILURES_PER_REQUEST = 6;
const TAINT_FRESH_MILLI = 45;
const TAINT_OLD_MILLI = 15;
const TAINT_REQUESTS = 2;

function loadBrief(): Brief {
  const file = process.env.MAKEFUN_PROMPT_FILE;
  if (file === undefined || file === '') return { providers: [], ratings: [], manifest: [] };
  try {
    return parseBrief(readFileSync(file, 'utf8'));
  } catch {
    return { providers: [], ratings: [], manifest: [] };
  }
}

const brief = loadBrief();
const catalog = buildCatalog(brief);
const catalogText = JSON.stringify(catalog);
const envUrl = (process.env.MAKEFUN_ENV_URL ?? '').replace(/\/+$/, '');

const providerById = new Map<string, ProviderDoc>();
for (const provider of brief.providers) providerById.set(provider.id, provider);

// Routing state, all a function of the requests served so far: how many
// calls each provider has taken, and at which request it last refused.
const callsMade = new Map<string, number>();
const lastRefusalAt = new Map<string, number>();
let requestsServed = 0;

/**
 * The taint a provider carries into a request, in milli of rating. A
 * refusal on the immediately preceding request is fresh evidence the
 * provider is spent or out, and costs it heavily; a refusal two requests
 * back may have been one scheduled blip, and costs it only a little. The
 * taint is forgotten once two further requests have been handled, or at
 * once when the provider answers again.
 */
function taintPenalty(providerId: string, requestIndex: number): number {
  const at = lastRefusalAt.get(providerId);
  if (at === undefined) return 0;
  const age = requestIndex - at;
  if (age <= 1) return TAINT_FRESH_MILLI;
  if (age <= TAINT_REQUESTS) return TAINT_OLD_MILLI;
  return 0;
}

/**
 * The candidates for one request, best first. Only free models that meet
 * the request's requirements qualify. A request the client put in JSON
 * mode must ride a wire that can echo JSON mode back, so Anthropic-wire
 * providers sit those out: their wire has no such flag to echo.
 *
 * Order is rating first, docked by each provider's taint (see
 * taintPenalty): a fresh refusal costs a provider its place against
 * every candidate rated within 45 milli of it, an older one only within
 * 15, so a run of refusals cannot spend a request's whole retry budget
 * re-asking providers that just said no while a sound mid-table provider
 * waits below them. Equally placed candidates go to the provider called
 * least so far, then by provider and model id, so the order is fully
 * determined by the brief and the trial so far.
 */
function planCandidates(requires: Requires, jsonMode: boolean, requestIndex: number): RatingRow[] {
  let rows = brief.ratings.filter((row) => row.free && meetsRequirements(row, requires));
  if (jsonMode) {
    rows = rows.filter((row) => providerById.get(row.provider)?.wire !== 'anthropic');
  }
  if (rows.length === 0) {
    // Nothing legal can serve this request as specified; still answer
    // with the best free model available rather than not answering.
    rows = brief.ratings.filter((row) => row.free);
  }
  const placed = rows.map((row) => {
    const penalty = taintPenalty(row.provider, requestIndex);
    return {
      row,
      tainted: penalty > 0,
      score: row.rating_milli - penalty,
      calls: callsMade.get(row.provider) ?? 0,
    };
  });
  placed.sort((a, b) => {
    if (b.score !== a.score) return b.score - a.score;
    if (a.tainted !== b.tainted) return a.tainted ? 1 : -1;
    if (a.calls !== b.calls) return a.calls - b.calls;
    if (a.row.provider !== b.row.provider) return a.row.provider < b.row.provider ? -1 : 1;
    if (a.row.model !== b.row.model) return a.row.model < b.row.model ? -1 : 1;
    return 0;
  });
  return placed.map((entry) => entry.row);
}

/** The key a provider is dialed with: the brief's own, else the secret it names. */
function providerKey(provider: ProviderDoc): string {
  if (provider.key !== '') return provider.key;
  if (provider.key_env !== '') return process.env[provider.key_env] ?? '';
  return '';
}

/** The absolute URL of a provider: its path joined to the env, never to a client. */
function providerBase(provider: ProviderDoc): string {
  if (/^https?:\/\//.test(provider.path)) return provider.path.replace(/\/+$/, '');
  return envUrl + provider.path;
}

type CallOutcome =
  | { ok: true; answer: NormalizedAnswer }
  | { ok: false };

/**
 * One call to one provider with one model. Same-wire traffic passes the
 * client's body through untouched apart from the model and the stream
 * flag — providers are always dialed non-streaming, and the client's
 * stream is rendered here from the whole answer — so nothing the client
 * sent is rewritten or dropped. Cross-wire traffic goes through the
 * normalized form in lib/wire.ts.
 */
async function callProvider(
  provider: ProviderDoc,
  model: string,
  clientApi: ClientApi,
  rawBody: Json,
  norm: NormalizedRequest,
): Promise<CallOutcome> {
  const dial = { wire: provider.wire, auth: provider.auth, key: providerKey(provider) };
  const body: Json =
    provider.wire === clientApi
      ? { ...rawBody, model, stream: false }
      : buildProviderRequest(provider.wire, norm, model);
  const url = providerBase(provider) + providerPath(provider.wire, model);
  let response: Response;
  try {
    response = await fetch(url, {
      method: 'POST',
      headers: providerHeaders(dial),
      body: JSON.stringify(body),
    });
  } catch {
    return { ok: false };
  }
  if (!response.ok) {
    try {
      await response.text();
    } catch {
      // The refusal is already counted; its body carries nothing we need.
    }
    return { ok: false };
  }
  try {
    const value: unknown = await response.json();
    if (!isRecord(value)) return { ok: false };
    return { ok: true, answer: parseProviderAnswer(provider.wire, value) };
  } catch {
    return { ok: false };
  }
}

/**
 * What the request at this index needs. The client sends the workload in
 * manifest order, one request at a time, so the manifest row at the
 * request's own index is authoritative. Past the manifest's end — or with
 * no manifest at all — the requirements are read off the request itself.
 */
function manifestRequires(index: number, norm: NormalizedRequest): Requires {
  const row = brief.manifest[index];
  if (row !== undefined) return row.requires;
  const derived: Requires = {};
  if (norm.tools.length > 0) derived.tools = true;
  if (norm.turns.some((turn) => turn.images.length > 0)) derived.vision = true;
  if (norm.jsonMode) derived.json = true;
  return derived;
}

/** Serve one client request: plan, dial with failover, render the answer. */
async function serveRequest(
  clientApi: ClientApi,
  rawBody: Json,
): Promise<{ status: number; contentType: string; body: string }> {
  const index = requestsServed;
  const norm = clientApi === 'openai' ? parseOpenAIRequest(rawBody) : parseAnthropicRequest(rawBody);
  const requires = manifestRequires(index, norm);
  const candidates = planCandidates(requires, norm.jsonMode, index);
  let failures = 0;
  let answer: NormalizedAnswer | null = null;
  let servedModel = '';
  for (const candidate of candidates) {
    if (failures >= MAX_FAILURES_PER_REQUEST) break;
    const provider = providerById.get(candidate.provider);
    if (provider === undefined) continue;
    callsMade.set(provider.id, (callsMade.get(provider.id) ?? 0) + 1);
    const outcome = await callProvider(provider, candidate.model, clientApi, rawBody, norm);
    if (outcome.ok) {
      lastRefusalAt.delete(provider.id);
      answer = outcome.answer;
      servedModel = candidate.model;
      break;
    }
    failures += 1;
    lastRefusalAt.set(provider.id, index);
  }
  requestsServed += 1;
  if (answer === null) {
    const message = 'no free provider could serve this request';
    const error = clientApi === 'openai' ? openAIError(message) : anthropicError(message);
    return { status: 502, contentType: 'application/json', body: JSON.stringify(error) };
  }
  if (norm.stream) {
    const sse = clientApi === 'openai' ? renderOpenAISse(answer, servedModel) : renderAnthropicSse(answer, servedModel);
    return { status: 200, contentType: 'text/event-stream; charset=utf-8', body: sse };
  }
  const json = clientApi === 'openai' ? renderOpenAIJson(answer, servedModel) : renderAnthropicJson(answer, servedModel);
  return { status: 200, contentType: 'application/json', body: JSON.stringify(json) };
}

/** The free model ids of the catalog, each once, in catalog order. */
function listedModelIds(): string[] {
  const seen = new Set<string>();
  const ids: string[] = [];
  for (const id of freeModelIds(catalog)) {
    if (seen.has(id)) continue;
    seen.add(id);
    ids.push(id);
  }
  return ids;
}

/**
 * GET /v1/models, in the wire the caller's auth header names: an
 * `Authorization: Bearer` request gets the OpenAI model list, an
 * `x-api-key` request the Anthropic one. Both list exactly the free
 * model ids the catalog registers.
 */
function modelsReply(req: IncomingMessage): { status: number; contentType: string; body: string } {
  const ids = listedModelIds();
  if (req.headers['x-api-key'] !== undefined) {
    const body = {
      data: ids.map((id) => ({
        type: 'model',
        id,
        display_name: id,
        created_at: '2026-01-01T00:00:00Z',
      })),
      has_more: false,
    };
    return { status: 200, contentType: 'application/json', body: JSON.stringify(body) };
  }
  const body = {
    object: 'list',
    data: ids.map((id) => ({ id, object: 'model', created: 1767225600, owned_by: 'aggregator' })),
  };
  return { status: 200, contentType: 'application/json', body: JSON.stringify(body) };
}

function readBody(req: IncomingMessage): Promise<string> {
  return new Promise((resolve, reject) => {
    const chunks: Buffer[] = [];
    let size = 0;
    req.on('data', (chunk: Buffer) => {
      size += chunk.length;
      if (size <= 4 * 1024 * 1024) chunks.push(chunk);
    });
    req.on('end', () => resolve(Buffer.concat(chunks).toString('utf8')));
    req.on('error', reject);
  });
}

function send(res: ServerResponse, reply: { status: number; contentType: string; body: string }): void {
  res.writeHead(reply.status, {
    'content-type': reply.contentType,
    'content-length': String(Buffer.byteLength(reply.body)),
  });
  res.end(reply.body);
}

async function handle(req: IncomingMessage, res: ServerResponse): Promise<void> {
  const path = (req.url ?? '').split('?')[0];
  if (req.method === 'GET' && path === '/v1/models') {
    send(res, modelsReply(req));
    return;
  }
  if (req.method === 'POST' && (path === '/v1/chat/completions' || path === '/v1/messages')) {
    const clientApi: ClientApi = path === '/v1/messages' ? 'anthropic' : 'openai';
    let parsed: unknown = null;
    try {
      parsed = JSON.parse(await readBody(req));
    } catch {
      parsed = null;
    }
    if (!isRecord(parsed)) {
      const error = clientApi === 'openai'
        ? openAIError('the request body is not one JSON object', 'invalid_request')
        : anthropicError('the request body is not one JSON object');
      send(res, { status: 400, contentType: 'application/json', body: JSON.stringify(error) });
      return;
    }
    send(res, await serveRequest(clientApi, parsed));
    return;
  }
  send(res, {
    status: 404,
    contentType: 'application/json',
    body: JSON.stringify(openAIError(`no route ${req.method ?? ''} ${path}`, 'not_found')),
  });
}

function sleep(ms: number): Promise<void> {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

/** Register this server's port and catalog with the env, retrying briefly. */
async function register(port: number): Promise<void> {
  const body = JSON.stringify({ protocol_version: 1, port, catalog });
  for (let attempt = 0; attempt < 10; attempt += 1) {
    try {
      const response = await fetch(`${envUrl}/register`, {
        method: 'POST',
        headers: { 'content-type': 'application/json' },
        body,
      });
      await response.text();
      if (response.ok) return;
    } catch {
      // The env may still be coming up; the next attempt decides.
    }
    await sleep(300);
  }
}

/** Poll the env's /work until it says the workload is done, then exit 0. */
async function pollWorkUntilDone(): Promise<void> {
  for (;;) {
    try {
      const response = await fetch(`${envUrl}/work`);
      if (response.ok) {
        const value: unknown = await response.json();
        if (isRecord(value) && value.done === true) return;
      }
    } catch {
      // A failed poll says nothing about the workload; poll again.
    }
    await sleep(200);
  }
}

async function main(): Promise<void> {
  if (process.argv.includes('--list')) {
    process.stdout.write(`${catalogText}\n`);
    return;
  }
  const server = createServer((req, res) => {
    void handle(req, res).catch(() => {
      try {
        send(res, {
          status: 500,
          contentType: 'application/json',
          body: JSON.stringify(openAIError('internal error', 'internal')),
        });
      } catch {
        // The client is gone; there is nothing left to answer.
      }
    });
  });
  await new Promise<void>((resolve) => server.listen(0, '0.0.0.0', resolve));
  const address = server.address();
  const port = address !== null && typeof address === 'object' ? address.port : 0;
  if (envUrl !== '') {
    await register(port);
    await pollWorkUntilDone();
    server.close();
    process.exit(0);
  }
}

void main();
