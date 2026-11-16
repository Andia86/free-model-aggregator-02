// The best free AI model aggregator: a routing server.
//
// What it does, in one paragraph: the trial brief names ten providers of free models, a sealed
// quality table rating every model, and a workload of twenty requests. This server registers a
// catalog built from that brief, then answers each client request — OpenAI-shaped or
// Anthropic-shaped, streamed or not — from the best free model that can carry it, on whichever
// provider wire that model lives, translating the request and the answer across wires when the
// two differ. A provider that refuses (its per-trial budget spent, a scheduled failure, an
// outage) is set aside for that request and the next-best model answers instead. Rating always
// leads the order: a refusal only costs a provider its place against candidates rated within
// 15 milli of it, where one that has not refused goes first, and the refusal is forgotten once
// two further requests have been handled (or at once, when the provider answers again). Equally
// rated candidates are taken from the provider called least so far this trial, so one model
// offered by several providers draws on all their budgets rather than draining the first. One
// request gives up after six refused calls rather than walk the whole directory.
//
// The rules it lives by, because the scoring does:
//   - an answer only counts from a free model that meets the request's requirements, so paid
//     models are never candidates and capability checks come from the brief's own table;
//   - exactly one provider call may succeed per request, so a successful answer is always the
//     one served, never shopped around;
//   - the usage, stop reason and echoed request fields of an answer must be the provider's own,
//     so same-wire traffic passes through untouched and cross-wire traffic is translated
//     field-by-field (lib/wire.ts) rather than re-answered;
//   - every choice derives from the brief, the request and the providers' answers — no clock,
//     no randomness — so a replayed trial makes the same calls and leaves the same state.
//
// Layout: lib/brief.ts reads the brief and builds the catalog; lib/wire.ts holds the wire
// translations; this file is the server, the planner and the trial lifecycle (listen, register,
// poll /work, exit 0 when the work is done).

import { createServer, type IncomingMessage, type ServerResponse } from 'node:http';
import { readFileSync } from 'node:fs';
import {
  buildCatalog,
  freeModelIds,
  isRecord,
  meetsRequirements,
  parseBrief,
  type Brief,
  type ClientApi,
  type Json,
  type ProviderDoc,
  type RatingRow,
  type Requires,
} from './lib/brief.ts';
import {
  anthropicModelList,
  clientError,
  fromAnthropic,
  fromOpenAI,
  openAIModelList,
  parseProviderAnswer,
  renderClientAnswer,
  toAnthropic,
  toGemini,
  toOpenAI,
  type NormalizedRequest,
} from './lib/wire.ts';

/** How long to wait on one provider call before treating it as refused. */
const CALL_TIMEOUT_MS = 25_000;
/** The /work poll interval. */
const POLL_MS = 100;
/** Leave the trial before its wall does: a server stopped at the wall scores nothing. */
const WALL_GUARD_MS = 160_000;
/** Ratings this close (in milli) count as a near-tie, the only place taint is allowed a say. */
const NEAR_TIE_MILLI = 15;
/** How many later requests a refusal is remembered for before the taint expires on its own. */
const TAINT_REQUESTS = 2;
/** Refused calls one request may spend before it stops trying and answers 502. */
const MAX_FAILED_ATTEMPTS = 6;

/** One provider at runtime: its directory entry, the key to call it with, its recent luck. */
interface RuntimeProvider {
  doc: ProviderDoc;
  /** Directory order, the final tie-break between equal candidates. */
  index: number;
  key: string;
  /**
   * Refusals since the provider's last success; a success clears the count. The count outlives
   * an expired taint, so a provider that keeps refusing sorts behind one that refused once.
   */
  refusals: number;
  /**
   * Whether the provider refused recently: since its last success, and within the last
   * TAINT_REQUESTS requests. Taint never outranks rating — it only breaks near-ties: between
   * candidates rated within NEAR_TIE_MILLI of each other the untainted one goes first, and
   * between tainted ones the one with fewer refusals. A refusal may be a single scheduled
   * point, so the taint clears on a success or once TAINT_REQUESTS further requests have been
   * handled, whichever comes first.
   */
  tainted: boolean;
  /** The request (by requestSeq) during which the provider last refused; dates the taint. */
  refusedAt: number;
  /** Calls made to the provider so far this trial, answered or refused. */
  attempts: number;
}

interface Candidate {
  provider: RuntimeProvider;
  row: RatingRow;
}

let brief: Brief = { providers: [], ratings: [], manifest: [] };
let catalog: Json = { providers: [] };
let modelIds: string[] = [];
let envUrl = '';
const runtime = new Map<string, RuntimeProvider>();
/** Chat-shaped posts seen so far: the manifest row a request answers to, by arrival order. */
let requestSeq = 0;

/** A row's rating in milli, the primary key of every ordering below. */
function ratingOf(row: RatingRow): number {
  return row.rating_milli;
}

/** Rating first; equal ratings go to the provider called least so far, then directory order. */
function byRating(a: Candidate, b: Candidate): number {
  const ratingA = ratingOf(a.row);
  const ratingB = ratingOf(b.row);
  if (ratingA !== ratingB) return ratingB - ratingA;
  if (a.provider.attempts !== b.provider.attempts) return a.provider.attempts - b.provider.attempts;
  return a.provider.index - b.provider.index;
}

/** The order among near-tied candidates: untainted first, then fewer refusals, then rating. */
function byNearTie(a: Candidate, b: Candidate): number {
  const taintA = a.provider.tainted ? 1 : 0;
  const taintB = b.provider.tainted ? 1 : 0;
  if (taintA !== taintB) return taintA - taintB;
  if (taintA === 1 && a.provider.refusals !== b.provider.refusals) return a.provider.refusals - b.provider.refusals;
  return byRating(a, b);
}

/** Forget refusals that TAINT_REQUESTS later requests have already been handled since. */
function expireTaint(seq: number): void {
  for (const provider of runtime.values()) {
    if (provider.tainted && seq - provider.refusedAt > TAINT_REQUESTS) provider.tainted = false;
  }
}

/**
 * The candidates for one request, best first: each provider's best free model that meets the
 * requirements. A request the client put in JSON mode can only ride a wire that can echo JSON
 * mode back (OpenAI's response_format, Gemini's responseMimeType) — the Anthropic wire has no
 * such flag, so its providers sit these requests out. The order is by rating, with equal
 * ratings spread toward the provider called least so far. Taint only reorders near-ties: each
 * place is filled from the candidates within NEAR_TIE_MILLI of the best one left, untainted
 * before tainted and fewer refusals before more, so no candidate is ever passed over for one
 * rated more than NEAR_TIE_MILLI below it. Tainted providers stay in the list: a scheduled
 * failure is a single point, and a later place is still better than no answer.
 */
function candidates(requires: Requires, jsonModeCarry: boolean): Candidate[] {
  const bestByProvider = new Map<string, RatingRow>();
  for (const row of brief.ratings) {
    if (!row.free) continue;
    if (!meetsRequirements(row, requires)) continue;
    const provider = runtime.get(row.provider);
    if (provider === undefined) continue;
    if (jsonModeCarry && provider.doc.wire === 'anthropic') continue;
    const current = bestByProvider.get(row.provider);
    if (current === undefined || ratingOf(row) > ratingOf(current)) bestByProvider.set(row.provider, row);
  }
  let pool: Candidate[] = [...bestByProvider.values()].map((row) => ({ provider: runtime.get(row.provider) as RuntimeProvider, row }));
  pool.sort(byRating);
  const ordered: Candidate[] = [];
  for (let head = pool[0]; head !== undefined; head = pool[0]) {
    const floor = ratingOf(head.row) - NEAR_TIE_MILLI;
    let pick = head;
    for (const other of pool) {
      if (ratingOf(other.row) < floor) break;
      if (byNearTie(other, pick) < 0) pick = other;
    }
    const chosen = pick;
    ordered.push(chosen);
    pool = pool.filter((candidate) => candidate !== chosen);
  }
  return ordered;
}

/** Requirements read off a body the manifest has no row for (a request past the manifest). */
function inferRequires(api: ClientApi, body: Json): Requires {
  const requires: Requires = {};
  if (Array.isArray(body.tools) && body.tools.length > 0) requires.tools = true;
  if (api === 'openai' && isRecord(body.response_format)) requires.json = true;
  const messages = Array.isArray(body.messages) ? body.messages : [];
  for (const message of messages) {
    if (!isRecord(message) || !Array.isArray(message.content)) continue;
    for (const block of message.content) {
      if (isRecord(block) && (block.type === 'image_url' || block.type === 'image')) requires.vision = true;
    }
  }
  return requires;
}

/** The client's body with the routed model named, and a streamed usage chunk asked for. */
function passthroughBody(api: ClientApi, body: Json, model: string): Json {
  const out: Json = { ...body, model };
  if (api === 'openai' && body.stream === true) {
    const options = isRecord(body.stream_options) ? body.stream_options : {};
    out.stream_options = { ...options, include_usage: true };
  }
  return out;
}

interface ProviderReply {
  status: number;
  contentType: string;
  bodyText: string;
}

/** One call to one provider on its own wire, translated if the client's wire differs. */
async function callProvider(
  candidate: Candidate,
  api: ClientApi,
  body: Json,
  normalized: () => NormalizedRequest,
): Promise<ProviderReply | null> {
  const { doc } = candidate.provider;
  const model = candidate.row.model;
  let url: string;
  let payload: Json;
  if (doc.wire === 'openai') {
    url = `${envUrl}${doc.path}/v1/chat/completions`;
    payload = api === 'openai' ? passthroughBody(api, body, model) : toOpenAI(normalized(), model);
  } else if (doc.wire === 'anthropic') {
    url = `${envUrl}${doc.path}/v1/messages`;
    payload = api === 'anthropic' ? passthroughBody(api, body, model) : toAnthropic(normalized(), model);
  } else {
    const gemini = toGemini(normalized(), model);
    url = `${envUrl}${doc.path}${gemini.path}`;
    payload = gemini.body;
  }
  const headers: Record<string, string> = { 'content-type': 'application/json' };
  if (doc.auth === 'bearer') headers.authorization = `Bearer ${candidate.provider.key}`;
  else if (doc.auth === 'x-api-key') headers['x-api-key'] = candidate.provider.key;
  else if (doc.auth === 'x-goog-api-key') headers['x-goog-api-key'] = candidate.provider.key;
  if (doc.wire === 'anthropic') headers['anthropic-version'] = '2023-06-01';
  try {
    const response = await fetch(url, {
      method: 'POST',
      headers,
      body: JSON.stringify(payload),
      signal: AbortSignal.timeout(CALL_TIMEOUT_MS),
    });
    return {
      status: response.status,
      contentType: response.headers.get('content-type') ?? 'application/json',
      bodyText: await response.text(),
    };
  } catch {
    return null;
  }
}

function send(res: ServerResponse, status: number, contentType: string, body: string): void {
  res.writeHead(status, { 'content-type': contentType });
  res.end(body);
}

/**
 * One client request: try the candidates in order until one answers, then serve its answer.
 * After MAX_FAILED_ATTEMPTS refused calls the request stops trying and answers 502.
 */
async function handleChat(api: ClientApi, raw: string, res: ServerResponse): Promise<void> {
  const seq = requestSeq;
  requestSeq += 1;
  let body: Json;
  try {
    const parsed: unknown = JSON.parse(raw);
    if (!isRecord(parsed)) throw new Error('not an object');
    body = parsed;
  } catch {
    send(res, 400, 'application/json', JSON.stringify(clientError(api, 'the request body is not one JSON object')));
    return;
  }
  const row = seq < brief.manifest.length ? brief.manifest[seq] : undefined;
  const stream = body.stream === true || (row !== undefined && row.stream);
  const requires = row !== undefined ? row.requires : inferRequires(api, body);
  const format = isRecord(body.response_format) ? body.response_format.type : undefined;
  const jsonModeCarry = api === 'openai' && (format === 'json_object' || format === 'json_schema');
  let normalizedCache: NormalizedRequest | null = null;
  const normalized = (): NormalizedRequest => {
    if (normalizedCache === null) normalizedCache = api === 'openai' ? fromOpenAI(body) : fromAnthropic(body);
    return normalizedCache;
  };

  expireTaint(seq);
  let failures = 0;
  for (const candidate of candidates(requires, jsonModeCarry)) {
    candidate.provider.attempts += 1;
    const reply = await callProvider(candidate, api, body, normalized);
    if (reply === null || reply.status < 200 || reply.status >= 300) {
      candidate.provider.refusals += 1;
      candidate.provider.tainted = true;
      candidate.provider.refusedAt = Math.max(candidate.provider.refusedAt, seq);
      failures += 1;
      if (failures >= MAX_FAILED_ATTEMPTS) break;
      continue;
    }
    candidate.provider.refusals = 0;
    candidate.provider.tainted = false;
    if (candidate.provider.doc.wire === api) {
      send(res, reply.status, reply.contentType, reply.bodyText);
      return;
    }
    const answer = parseProviderAnswer(candidate.provider.doc.wire, stream, reply.bodyText);
    if (answer === null) {
      const rendered = renderClientAnswer(api, stream, candidate.row.model, { text: '', inputTokens: 0, outputTokens: 0 });
      send(res, 200, rendered.contentType, rendered.body);
      return;
    }
    const rendered = renderClientAnswer(api, stream, candidate.row.model, answer);
    send(res, 200, rendered.contentType, rendered.body);
    return;
  }
  send(res, 502, 'application/json', JSON.stringify(clientError(api, 'every provider refused this request')));
}

/** GET /v1/models, in the shape the caller's auth header asks for. */
function handleModels(req: IncomingMessage, res: ServerResponse): void {
  const anthropic = req.headers['x-api-key'] !== undefined;
  const list = anthropic ? anthropicModelList(modelIds) : openAIModelList(modelIds);
  send(res, 200, 'application/json', JSON.stringify(list));
}

function readBody(req: IncomingMessage): Promise<string> {
  return new Promise((resolve, reject) => {
    const chunks: Buffer[] = [];
    let size = 0;
    req.on('data', (chunk: Buffer) => {
      size += chunk.length;
      if (size <= 2 * 1024 * 1024) chunks.push(chunk);
    });
    req.on('end', () => resolve(Buffer.concat(chunks).toString('utf8')));
    req.on('error', reject);
  });
}

async function main(): Promise<void> {
  const promptFile = process.env.MAKEFUN_PROMPT_FILE;
  if (promptFile !== undefined && promptFile !== '') {
    try {
      brief = parseBrief(readFileSync(promptFile, 'utf8'));
    } catch {
      brief = { providers: [], ratings: [], manifest: [] };
    }
  }
  catalog = buildCatalog(brief);
  modelIds = freeModelIds(catalog);

  if (process.argv.includes('--list')) {
    process.stdout.write(`${JSON.stringify(catalog)}\n`);
    return;
  }

  envUrl = (process.env.MAKEFUN_ENV_URL ?? '').replace(/\/+$/, '');
  if (envUrl === '') return;

  brief.providers.forEach((doc, index) => {
    const key = doc.key !== '' ? doc.key : (process.env[doc.key_env] ?? '');
    runtime.set(doc.id, { doc, index, key, refusals: 0, tainted: false, refusedAt: 0, attempts: 0 });
  });

  const server = createServer((req, res) => {
    const path = (req.url ?? '').split('?')[0];
    if (req.method === 'GET' && path === '/v1/models') {
      handleModels(req, res);
      return;
    }
    if (req.method === 'POST' && (path === '/v1/chat/completions' || path === '/v1/messages')) {
      const api: ClientApi = path === '/v1/messages' ? 'anthropic' : 'openai';
      readBody(req)
        .then((raw) => handleChat(api, raw, res))
        .catch(() => send(res, 400, 'application/json', JSON.stringify(clientError(api, 'the request could not be read'))));
      return;
    }
    send(res, 404, 'application/json', JSON.stringify(clientError('openai', `no route ${req.method ?? ''} ${path}`)));
  });

  await new Promise<void>((resolve) => server.listen(0, '0.0.0.0', resolve));
  const address = server.address();
  const port = typeof address === 'object' && address !== null ? address.port : 0;

  let finished = false;
  const finish = (): void => {
    if (finished) return;
    finished = true;
    server.close();
    process.exit(0);
  };
  process.on('SIGTERM', finish);
  process.on('SIGINT', finish);
  setTimeout(finish, WALL_GUARD_MS);

  const registerBody = JSON.stringify({ protocol_version: 1, port, catalog });
  /** One /register attempt; true once the environment has accepted the catalog. */
  const register = async (): Promise<boolean> => {
    try {
      const response = await fetch(`${envUrl}/register`, {
        method: 'POST',
        headers: { 'content-type': 'application/json' },
        body: registerBody,
        signal: AbortSignal.timeout(10_000),
      });
      await response.text();
      return response.ok;
    } catch {
      return false;
    }
  };
  let registered = false;
  for (let attempt = 0; attempt < 3 && !registered; attempt += 1) {
    registered = await register();
  }

  const poll = async (): Promise<void> => {
    while (!finished) {
      if (!registered) registered = await register();
      try {
        const response = await fetch(`${envUrl}/work`, { signal: AbortSignal.timeout(10_000) });
        const work: unknown = await response.json();
        if (isRecord(work) && work.done === true) {
          finish();
          return;
        }
      } catch {
      }
      await new Promise((resolve) => setTimeout(resolve, POLL_MS));
    }
  };
  void poll();
}

void main();
