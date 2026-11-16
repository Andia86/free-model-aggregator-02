// Wire translations for the aggregator.
//
// Clients speak one of two APIs — OpenAI Chat Completions or Anthropic
// Messages — and providers speak one of three wires: OpenAI, Anthropic or
// Gemini. Every request is first read into one normalized form, then written
// out again in the chosen provider's wire; every provider answer is read back
// into a normalized answer and rendered in the client's wire, as JSON or as
// server-sent events, as the client asked.
//
// The normalized form keeps exactly the parts of a request a provider can
// echo back: the system text, the conversation turns with their images, the
// tool definitions, the output cap, the stop sequences and the JSON-mode
// flag. Keeping those intact across a wire change is what lets an answer
// stay a faithful answer to the request that was actually sent.

import { isRecord } from './brief.ts';
import type { Json, Wire } from './brief.ts';

/** An image carried by a user turn: inline bytes, or a URL the provider reads. */
export type ImageRef =
  | { kind: 'base64'; mediaType: string; data: string }
  | { kind: 'url'; url: string };

/** One conversation turn in the normalized form. */
export interface Turn {
  role: 'user' | 'assistant';
  text: string;
  images: ImageRef[];
}

/** One tool definition, named once and rendered per wire. */
export interface ToolDef {
  name: string;
  description: string;
  parameters: Json;
}

/** A client request, read into the normalized form. */
export interface NormalizedRequest {
  systemText: string;
  turns: Turn[];
  tools: ToolDef[];
  maxTokens: number | null;
  stops: string[];
  jsonMode: boolean;
  stream: boolean;
}

/** A provider's answer, read into the normalized form. */
export interface NormalizedAnswer {
  text: string;
  stop: string | null;
  inputTokens: number;
  outputTokens: number;
}

/** How to dial one provider: its wire, its auth scheme and its key. */
export interface Dial {
  wire: Wire;
  auth: string;
  key: string;
}

function numberOrNull(v: unknown): number | null {
  return typeof v === 'number' && Number.isFinite(v) ? v : null;
}

function stringList(v: unknown): string[] {
  if (typeof v === 'string') return [v];
  if (!Array.isArray(v)) return [];
  return v.filter((x): x is string => typeof x === 'string');
}

/**
 * The text of a message content: a plain string is itself; a block list is
 * its text blocks joined with newlines. Both client wires read content this
 * way, so the normalized text is the text either wire would report.
 */
function textOfContent(content: unknown): string {
  if (typeof content === 'string') return content;
  if (!Array.isArray(content)) return '';
  const parts: string[] = [];
  for (const block of content) {
    if (isRecord(block) && block.type === 'text' && typeof block.text === 'string') {
      parts.push(block.text);
    }
  }
  return parts.join('\n');
}

/** Split a data URI into its media type and base64 payload, or null. */
function splitDataUri(url: string): { mediaType: string; data: string } | null {
  const match = /^data:([^;,]*)(?:;[^,]*)?;base64,(.*)$/s.exec(url);
  if (match === null) return null;
  return { mediaType: match[1] === '' ? 'image/png' : match[1], data: match[2] };
}

/** The images of one OpenAI user message, in the order they appear. */
function openAIImages(content: unknown): ImageRef[] {
  const images: ImageRef[] = [];
  if (!Array.isArray(content)) return images;
  for (const block of content) {
    if (!isRecord(block) || block.type !== 'image_url' || !isRecord(block.image_url)) continue;
    const url = block.image_url.url;
    if (typeof url !== 'string' || url === '') continue;
    const data = splitDataUri(url);
    if (data !== null) images.push({ kind: 'base64', mediaType: data.mediaType, data: data.data });
    else images.push({ kind: 'url', url });
  }
  return images;
}

/** The images of one Anthropic user message, in the order they appear. */
function anthropicImages(content: unknown): ImageRef[] {
  const images: ImageRef[] = [];
  if (!Array.isArray(content)) return images;
  for (const block of content) {
    if (!isRecord(block) || block.type !== 'image' || !isRecord(block.source)) continue;
    const source = block.source;
    if (source.type === 'base64' && typeof source.data === 'string') {
      images.push({
        kind: 'base64',
        mediaType: typeof source.media_type === 'string' ? source.media_type : 'image/png',
        data: source.data,
      });
    } else if (source.type === 'url' && typeof source.url === 'string') {
      images.push({ kind: 'url', url: source.url });
    }
  }
  return images;
}

/** Read an OpenAI Chat Completions request body into the normalized form. */
export function parseOpenAIRequest(body: Json): NormalizedRequest {
  const messages = Array.isArray(body.messages) ? body.messages.filter(isRecord) : [];
  const systemParts: string[] = [];
  const turns: Turn[] = [];
  for (const message of messages) {
    if (message.role === 'system') {
      const text = textOfContent(message.content);
      if (text !== '') systemParts.push(text);
    } else if (message.role === 'user' || message.role === 'assistant') {
      turns.push({
        role: message.role,
        text: textOfContent(message.content),
        images: message.role === 'user' ? openAIImages(message.content) : [],
      });
    }
  }
  const tools: ToolDef[] = [];
  if (Array.isArray(body.tools)) {
    for (const tool of body.tools) {
      if (!isRecord(tool) || !isRecord(tool.function)) continue;
      if (typeof tool.function.name !== 'string') continue;
      tools.push({
        name: tool.function.name,
        description: typeof tool.function.description === 'string' ? tool.function.description : '',
        parameters: isRecord(tool.function.parameters) ? tool.function.parameters : {},
      });
    }
  }
  const format = isRecord(body.response_format) ? body.response_format.type : undefined;
  return {
    systemText: systemParts.join('\n'),
    turns,
    tools,
    maxTokens: numberOrNull(body.max_tokens) ?? numberOrNull(body.max_completion_tokens),
    stops: stringList(body.stop),
    jsonMode: format === 'json_object' || format === 'json_schema',
    stream: body.stream === true,
  };
}

/** Read an Anthropic Messages request body into the normalized form. */
export function parseAnthropicRequest(body: Json): NormalizedRequest {
  const messages = Array.isArray(body.messages) ? body.messages.filter(isRecord) : [];
  const turns: Turn[] = [];
  for (const message of messages) {
    if (message.role !== 'user' && message.role !== 'assistant') continue;
    turns.push({
      role: message.role,
      text: textOfContent(message.content),
      images: message.role === 'user' ? anthropicImages(message.content) : [],
    });
  }
  const tools: ToolDef[] = [];
  if (Array.isArray(body.tools)) {
    for (const tool of body.tools) {
      if (!isRecord(tool) || typeof tool.name !== 'string') continue;
      tools.push({
        name: tool.name,
        description: typeof tool.description === 'string' ? tool.description : '',
        parameters: isRecord(tool.input_schema) ? tool.input_schema : {},
      });
    }
  }
  return {
    systemText: textOfContent(body.system),
    turns,
    tools,
    maxTokens: numberOrNull(body.max_tokens),
    stops: stringList(body.stop_sequences),
    jsonMode: false,
    stream: body.stream === true,
  };
}

/** One user turn as OpenAI content: a plain string, or text and image blocks. */
function openAITurnContent(turn: Turn): unknown {
  if (turn.images.length === 0) return turn.text;
  const blocks: unknown[] = [];
  if (turn.text !== '') blocks.push({ type: 'text', text: turn.text });
  for (const image of turn.images) {
    const url = image.kind === 'base64' ? `data:${image.mediaType};base64,${image.data}` : image.url;
    blocks.push({ type: 'image_url', image_url: { url } });
  }
  return blocks;
}

/** Write the normalized form out as an OpenAI Chat Completions request body. */
export function buildOpenAIRequest(norm: NormalizedRequest, model: string): Json {
  const messages: unknown[] = [];
  if (norm.systemText !== '') messages.push({ role: 'system', content: norm.systemText });
  for (const turn of norm.turns) {
    messages.push({ role: turn.role, content: openAITurnContent(turn) });
  }
  const body: Json = { model, messages, stream: false };
  if (norm.maxTokens !== null) body.max_tokens = norm.maxTokens;
  if (norm.stops.length > 0) body.stop = [...norm.stops];
  if (norm.tools.length > 0) {
    body.tools = norm.tools.map((tool) => ({
      type: 'function',
      function: { name: tool.name, description: tool.description, parameters: tool.parameters },
    }));
  }
  if (norm.jsonMode) body.response_format = { type: 'json_object' };
  return body;
}

/** One user turn as Anthropic content: a plain string, or text and image blocks. */
function anthropicTurnContent(turn: Turn): unknown {
  if (turn.images.length === 0) return turn.text;
  const blocks: unknown[] = [];
  if (turn.text !== '') blocks.push({ type: 'text', text: turn.text });
  for (const image of turn.images) {
    if (image.kind === 'base64') {
      blocks.push({
        type: 'image',
        source: { type: 'base64', media_type: image.mediaType, data: image.data },
      });
    } else {
      blocks.push({ type: 'image', source: { type: 'url', url: image.url } });
    }
  }
  return blocks;
}

/** Write the normalized form out as an Anthropic Messages request body. */
export function buildAnthropicRequest(norm: NormalizedRequest, model: string): Json {
  const messages = norm.turns.map((turn) => ({
    role: turn.role,
    content: turn.role === 'user' ? anthropicTurnContent(turn) : turn.text,
  }));
  const body: Json = { model, messages, stream: false };
  if (norm.systemText !== '') body.system = norm.systemText;
  if (norm.maxTokens !== null) body.max_tokens = norm.maxTokens;
  if (norm.stops.length > 0) body.stop_sequences = [...norm.stops];
  if (norm.tools.length > 0) {
    body.tools = norm.tools.map((tool) => ({
      name: tool.name,
      description: tool.description,
      input_schema: tool.parameters,
    }));
  }
  return body;
}

/** Write the normalized form out as a Gemini generateContent request body. */
export function buildGeminiRequest(norm: NormalizedRequest, model: string): Json {
  void model;
  const contents = norm.turns.map((turn) => {
    const parts: unknown[] = [];
    if (turn.text !== '') parts.push({ text: turn.text });
    for (const image of turn.images) {
      if (image.kind === 'base64') {
        parts.push({ inlineData: { mimeType: image.mediaType, data: image.data } });
      } else {
        parts.push({ fileData: { fileUri: image.url } });
      }
    }
    if (parts.length === 0) parts.push({ text: '' });
    return { role: turn.role === 'assistant' ? 'model' : 'user', parts };
  });
  const body: Json = { contents };
  if (norm.systemText !== '') {
    body.systemInstruction = { parts: [{ text: norm.systemText }] };
  }
  if (norm.tools.length > 0) {
    body.tools = [
      {
        functionDeclarations: norm.tools.map((tool) => ({
          name: tool.name,
          description: tool.description,
          parameters: tool.parameters,
        })),
      },
    ];
  }
  const config: Json = {};
  if (norm.maxTokens !== null) config.maxOutputTokens = norm.maxTokens;
  if (norm.stops.length > 0) config.stopSequences = [...norm.stops];
  if (norm.jsonMode) config.responseMimeType = 'application/json';
  if (Object.keys(config).length > 0) body.generationConfig = config;
  return body;
}

/** The request body for a provider's wire, from the normalized form. */
export function buildProviderRequest(wire: Wire, norm: NormalizedRequest, model: string): Json {
  if (wire === 'anthropic') return buildAnthropicRequest(norm, model);
  if (wire === 'gemini') return buildGeminiRequest(norm, model);
  return buildOpenAIRequest(norm, model);
}

/** The path of a provider call, relative to the provider's base URL. */
export function providerPath(wire: Wire, model: string): string {
  if (wire === 'anthropic') return '/v1/messages';
  if (wire === 'gemini') return `/v1beta/models/${encodeURIComponent(model)}:generateContent`;
  return '/v1/chat/completions';
}

/**
 * The headers of a provider call. The auth scheme — not the wire — decides
 * which header carries the key; a provider whose scheme is `none` takes no
 * key at all. Anthropic-wire calls also carry the anthropic-version header
 * the real API requires.
 */
export function providerHeaders(dial: Dial): Record<string, string> {
  const headers: Record<string, string> = { 'content-type': 'application/json' };
  if (dial.auth === 'bearer') headers.authorization = `Bearer ${dial.key}`;
  else if (dial.auth === 'x-api-key') headers['x-api-key'] = dial.key;
  else if (dial.auth === 'x-goog-api-key') headers['x-goog-api-key'] = dial.key;
  if (dial.wire === 'anthropic') headers['anthropic-version'] = '2023-06-01';
  return headers;
}

/** Read an OpenAI-wire provider answer into the normalized form. */
export function parseOpenAIAnswer(value: Json): NormalizedAnswer {
  const choices = Array.isArray(value.choices) ? value.choices : [];
  const choice = choices.length > 0 && isRecord(choices[0]) ? choices[0] : null;
  const message = choice !== null && isRecord(choice.message) ? choice.message : null;
  const usage = isRecord(value.usage) ? value.usage : {};
  return {
    text: message === null ? '' : textOfContent(message.content),
    stop: choice !== null && typeof choice.finish_reason === 'string' ? choice.finish_reason : null,
    inputTokens: numberOrNull(usage.prompt_tokens) ?? 0,
    outputTokens: numberOrNull(usage.completion_tokens) ?? 0,
  };
}

/** Read an Anthropic-wire provider answer into the normalized form. */
export function parseAnthropicAnswer(value: Json): NormalizedAnswer {
  const blocks = Array.isArray(value.content) ? value.content : [];
  const parts: string[] = [];
  for (const block of blocks) {
    if (isRecord(block) && block.type === 'text' && typeof block.text === 'string') {
      parts.push(block.text);
    }
  }
  const usage = isRecord(value.usage) ? value.usage : {};
  return {
    text: parts.join(''),
    stop: typeof value.stop_reason === 'string' ? value.stop_reason : null,
    inputTokens: numberOrNull(usage.input_tokens) ?? 0,
    outputTokens: numberOrNull(usage.output_tokens) ?? 0,
  };
}

/** Read a Gemini-wire provider answer into the normalized form. */
export function parseGeminiAnswer(value: Json): NormalizedAnswer {
  const candidates = Array.isArray(value.candidates) ? value.candidates : [];
  const candidate = candidates.length > 0 && isRecord(candidates[0]) ? candidates[0] : null;
  const content = candidate !== null && isRecord(candidate.content) ? candidate.content : null;
  const rawParts = content !== null && Array.isArray(content.parts) ? content.parts : [];
  const parts: string[] = [];
  for (const part of rawParts) {
    if (isRecord(part) && typeof part.text === 'string') parts.push(part.text);
  }
  const usage = isRecord(value.usageMetadata) ? value.usageMetadata : {};
  return {
    text: parts.join(''),
    stop: candidate !== null && typeof candidate.finishReason === 'string' ? candidate.finishReason : null,
    inputTokens: numberOrNull(usage.promptTokenCount) ?? 0,
    outputTokens: numberOrNull(usage.candidatesTokenCount) ?? 0,
  };
}

/** Read a provider answer of any wire into the normalized form. */
export function parseProviderAnswer(wire: Wire, value: Json): NormalizedAnswer {
  if (wire === 'anthropic') return parseAnthropicAnswer(value);
  if (wire === 'gemini') return parseGeminiAnswer(value);
  return parseOpenAIAnswer(value);
}

// Stop reasons travel in families: every wire has a word for a plain end, a
// length end, a tool-call end and a refusal, and a faithful rendering maps a
// provider's word to the same family's word in the client's wire.
const STOP_FAMILIES: Record<string, number> = {
  stop: 0,
  end_turn: 0,
  STOP: 0,
  length: 1,
  max_tokens: 1,
  MAX_TOKENS: 1,
  tool_calls: 2,
  function_call: 2,
  tool_use: 2,
  content_filter: 3,
  refusal: 3,
  SAFETY: 3,
};

export function stopFamily(stop: string | null): number {
  if (stop === null) return 0;
  return STOP_FAMILIES[stop] ?? 0;
}

const OPENAI_STOPS = ['stop', 'length', 'tool_calls', 'content_filter'];
const ANTHROPIC_STOPS = ['end_turn', 'max_tokens', 'tool_use', 'refusal'];

export function openAIStop(stop: string | null): string {
  return OPENAI_STOPS[stopFamily(stop)];
}

export function anthropicStop(stop: string | null): string {
  return ANTHROPIC_STOPS[stopFamily(stop)];
}

// A fixed creation stamp: an answer's shape never depends on the wall clock.
const FIXED_CREATED = 1767225600;

/** Render a normalized answer as an OpenAI Chat Completions JSON body. */
export function renderOpenAIJson(answer: NormalizedAnswer, model: string): Json {
  return {
    id: 'chatcmpl-aggregator',
    object: 'chat.completion',
    created: FIXED_CREATED,
    model,
    choices: [
      {
        index: 0,
        message: { role: 'assistant', content: answer.text },
        finish_reason: openAIStop(answer.stop),
        logprobs: null,
      },
    ],
    usage: {
      prompt_tokens: answer.inputTokens,
      completion_tokens: answer.outputTokens,
      total_tokens: answer.inputTokens + answer.outputTokens,
    },
  };
}

/** Render a normalized answer as an Anthropic Messages JSON body. */
export function renderAnthropicJson(answer: NormalizedAnswer, model: string): Json {
  return {
    id: 'msg_aggregator',
    type: 'message',
    role: 'assistant',
    model,
    content: [{ type: 'text', text: answer.text }],
    stop_reason: anthropicStop(answer.stop),
    stop_sequence: null,
    usage: { input_tokens: answer.inputTokens, output_tokens: answer.outputTokens },
  };
}

/** Split an answer text into stream pieces of a steady size. */
function splitText(text: string): string[] {
  const pieces: string[] = [];
  for (let at = 0; at < text.length; at += 24) pieces.push(text.slice(at, at + 24));
  return pieces.length === 0 ? [''] : pieces;
}

function sseData(value: unknown): string {
  return `data: ${JSON.stringify(value)}\n\n`;
}

function sseEvent(event: string, value: unknown): string {
  return `event: ${event}\n${sseData(value)}`;
}

/**
 * Render a normalized answer as an OpenAI SSE stream: a role chunk, the
 * content pieces, a finish chunk, a usage chunk (a stream only reports
 * usage when it carries one, so the usage always rides along), and [DONE].
 */
export function renderOpenAISse(answer: NormalizedAnswer, model: string): string {
  const chunk = (delta: Json, finish: string | null): Json => ({
    id: 'chatcmpl-aggregator',
    object: 'chat.completion.chunk',
    created: FIXED_CREATED,
    model,
    choices: [{ index: 0, delta, finish_reason: finish }],
  });
  let out = sseData(chunk({ role: 'assistant', content: '' }, null));
  for (const piece of splitText(answer.text)) {
    out += sseData(chunk({ content: piece }, null));
  }
  out += sseData(chunk({}, openAIStop(answer.stop)));
  out += sseData({
    id: 'chatcmpl-aggregator',
    object: 'chat.completion.chunk',
    created: FIXED_CREATED,
    model,
    choices: [],
    usage: {
      prompt_tokens: answer.inputTokens,
      completion_tokens: answer.outputTokens,
      total_tokens: answer.inputTokens + answer.outputTokens,
    },
  });
  out += 'data: [DONE]\n\n';
  return out;
}

/**
 * Render a normalized answer as an Anthropic SSE stream, in the event
 * order the Messages API streams: message_start, one text block opened,
 * its deltas, the block closed, message_delta with the stop reason and
 * the output usage, then message_stop.
 */
export function renderAnthropicSse(answer: NormalizedAnswer, model: string): string {
  let out = sseEvent('message_start', {
    type: 'message_start',
    message: {
      id: 'msg_aggregator',
      type: 'message',
      role: 'assistant',
      model,
      content: [],
      stop_reason: null,
      stop_sequence: null,
      usage: { input_tokens: answer.inputTokens, output_tokens: 0 },
    },
  });
  out += sseEvent('content_block_start', {
    type: 'content_block_start',
    index: 0,
    content_block: { type: 'text', text: '' },
  });
  for (const piece of splitText(answer.text)) {
    out += sseEvent('content_block_delta', {
      type: 'content_block_delta',
      index: 0,
      delta: { type: 'text_delta', text: piece },
    });
  }
  out += sseEvent('content_block_stop', { type: 'content_block_stop', index: 0 });
  out += sseEvent('message_delta', {
    type: 'message_delta',
    delta: { stop_reason: anthropicStop(answer.stop), stop_sequence: null },
    usage: { output_tokens: answer.outputTokens },
  });
  out += sseEvent('message_stop', { type: 'message_stop' });
  return out;
}

/** An error body in the OpenAI wire's envelope. */
export function openAIError(message: string, code = 'upstream_unavailable'): Json {
  return { error: { message, type: 'api_error', param: null, code } };
}

/** An error body in the Anthropic wire's envelope. */
export function anthropicError(message: string): Json {
  return { type: 'error', error: { type: 'api_error', message } };
}
