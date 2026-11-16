// The wire translations.
//
// A client speaks OpenAI or Anthropic; a provider speaks OpenAI, Anthropic or Gemini. When the
// two wires differ, the request is rewritten into the provider's wire and its answer rewritten
// back. The rewrite goes through one normalized form and is careful about exactly the things a
// provider echoes back about a request — the system text, the last user text, the tool names in
// order, the output cap, the stop sequences, the image bytes and the JSON-mode flag — because an
// answer is only worth anything if that echo matches what the client actually sent. Fields no
// echo covers (temperature, tool choice) are carried across in the obvious way.
//
// Same-wire traffic never comes here: main.ts passes it through untouched.
//
// Everything is a pure function: fixed timestamps, no clock, no randomness, so a replay of the
// same trial produces the same bytes.

import { isRecord, type ClientApi, type Json, type Wire } from './brief.ts';

/** A fixed epoch second for list and answer envelopes: nothing here reads the clock. */
export const FIXED_CREATED = 1767225600;
const FIXED_CREATED_AT = '2026-01-01T00:00:00Z';

/** One content part of a normalized turn. */
export type Part =
  | { type: 'text'; text: string }
  | { type: 'image'; mediaType: string; data: string }
  | { type: 'imageUrl'; url: string };

export interface ToolCallPart {
  id: string;
  name: string;
  argumentsJson: string;
}

export interface ToolResultPart {
  id: string;
  text: string;
}

export interface Turn {
  role: 'user' | 'assistant';
  parts: Part[];
  toolCalls: ToolCallPart[];
  toolResults: ToolResultPart[];
}

export interface ToolDef {
  name: string;
  description: string;
  schema: Json;
}

/** A client request with the wire stripped off: everything any provider wire needs from it. */
export interface NormalizedRequest {
  systemText: string;
  turns: Turn[];
  tools: ToolDef[];
  /** 'auto', 'required', or the name of the one tool the client insists on; null when silent. */
  toolChoice: string | null;
  maxTokens: number | null;
  stops: string[];
  temperature: number | null;
  /** The client asked for JSON mode on a wire that can say so (OpenAI response_format). */
  jsonMode: boolean;
  stream: boolean;
  /** Tool name by call id, so a later tool result can name its function on the Gemini wire. */
  toolNameById: Map<string, string>;
}

function numberOrNull(v: unknown): number | null {
  return typeof v === 'number' && Number.isFinite(v) ? v : null;
}

function stringList(v: unknown): string[] {
  if (typeof v === 'string') return [v];
  if (!Array.isArray(v)) return [];
  return v.filter((x): x is string => typeof x === 'string');
}

/** The text of a string or of a block array's text blocks, joined the way the providers join it. */
export function blocksText(content: unknown): string {
  if (typeof content === 'string') return content;
  if (!Array.isArray(content)) return '';
  const parts: string[] = [];
  for (const block of content) {
    if (isRecord(block) && block.type === 'text' && typeof block.text === 'string') parts.push(block.text);
  }
  return parts.join('\n');
}

function dataUrlOf(mediaType: string, data: string): string {
  return `data:${mediaType};base64,${data}`;
}

function parseDataUrl(url: string): { mediaType: string; data: string } | null {
  const m = /^data:([^;,]*);base64,(.*)$/s.exec(url);
  if (m === null) return null;
  return { mediaType: m[1] === '' ? 'image/png' : m[1], data: m[2] };
}

/** An OpenAI image_url value as a normalized part: inline bytes when the URL carries them. */
function openAIImagePart(url: unknown): Part | null {
  if (typeof url !== 'string' || url === '') return null;
  const inline = parseDataUrl(url);
  if (inline !== null) return { type: 'image', mediaType: inline.mediaType, data: inline.data };
  return { type: 'imageUrl', url };
}

/** The parts of one OpenAI message's content, text and images in order. */
function openAIParts(content: unknown): Part[] {
  if (typeof content === 'string') return content === '' ? [] : [{ type: 'text', text: content }];
  if (!Array.isArray(content)) return [];
  const parts: Part[] = [];
  for (const block of content) {
    if (!isRecord(block)) continue;
    if (block.type === 'text' && typeof block.text === 'string') parts.push({ type: 'text', text: block.text });
    else if (block.type === 'image_url' && isRecord(block.image_url)) {
      const part = openAIImagePart(block.image_url.url);
      if (part !== null) parts.push(part);
    }
  }
  return parts;
}

/** A tool_result block's content as plain text. */
function toolResultText(content: unknown): string {
  if (typeof content === 'string') return content;
  if (Array.isArray(content)) {
    const text = blocksText(content);
    return text !== '' ? text : JSON.stringify(content);
  }
  return content === undefined ? '' : JSON.stringify(content);
}

/** An OpenAI Chat Completions body in normalized form. */
export function fromOpenAI(body: Json): NormalizedRequest {
  const messages = Array.isArray(body.messages) ? body.messages : [];
  const systemParts: string[] = [];
  const turns: Turn[] = [];
  const toolNameById = new Map<string, string>();
  for (const raw of messages) {
    if (!isRecord(raw)) continue;
    if (raw.role === 'system') {
      const text = blocksText(raw.content);
      if (text !== '') systemParts.push(text);
      continue;
    }
    if (raw.role === 'tool') {
      turns.push({
        role: 'user',
        parts: [],
        toolCalls: [],
        toolResults: [
          {
            id: typeof raw.tool_call_id === 'string' ? raw.tool_call_id : '',
            text: typeof raw.content === 'string' ? raw.content : blocksText(raw.content),
          },
        ],
      });
      continue;
    }
    const turn: Turn = { role: raw.role === 'assistant' ? 'assistant' : 'user', parts: openAIParts(raw.content), toolCalls: [], toolResults: [] };
    if (Array.isArray(raw.tool_calls)) {
      for (const call of raw.tool_calls) {
        if (!isRecord(call) || !isRecord(call.function)) continue;
        const name = typeof call.function.name === 'string' ? call.function.name : '';
        const id = typeof call.id === 'string' ? call.id : '';
        turn.toolCalls.push({ id, name, argumentsJson: typeof call.function.arguments === 'string' ? call.function.arguments : '{}' });
        if (id !== '') toolNameById.set(id, name);
      }
    }
    turns.push(turn);
  }

  const tools: ToolDef[] = [];
  if (Array.isArray(body.tools)) {
    for (const raw of body.tools) {
      if (!isRecord(raw) || !isRecord(raw.function) || typeof raw.function.name !== 'string') continue;
      tools.push({
        name: raw.function.name,
        description: typeof raw.function.description === 'string' ? raw.function.description : '',
        schema: isRecord(raw.function.parameters) ? raw.function.parameters : { type: 'object', properties: {} },
      });
    }
  }

  let toolChoice: string | null = null;
  if (typeof body.tool_choice === 'string') toolChoice = body.tool_choice;
  else if (isRecord(body.tool_choice) && isRecord(body.tool_choice.function) && typeof body.tool_choice.function.name === 'string') {
    toolChoice = body.tool_choice.function.name;
  }

  const format = isRecord(body.response_format) ? body.response_format.type : undefined;
  return {
    systemText: systemParts.join('\n'),
    turns,
    tools,
    toolChoice,
    maxTokens: numberOrNull(body.max_tokens) ?? numberOrNull(body.max_completion_tokens),
    stops: stringList(body.stop),
    temperature: numberOrNull(body.temperature),
    jsonMode: format === 'json_object' || format === 'json_schema',
    stream: body.stream === true,
    toolNameById,
  };
}

/** An Anthropic Messages body in normalized form. */
export function fromAnthropic(body: Json): NormalizedRequest {
  const turns: Turn[] = [];
  const toolNameById = new Map<string, string>();
  const messages = Array.isArray(body.messages) ? body.messages : [];
  for (const raw of messages) {
    if (!isRecord(raw)) continue;
    const turn: Turn = { role: raw.role === 'assistant' ? 'assistant' : 'user', parts: [], toolCalls: [], toolResults: [] };
    if (typeof raw.content === 'string') {
      if (raw.content !== '') turn.parts.push({ type: 'text', text: raw.content });
    } else if (Array.isArray(raw.content)) {
      for (const block of raw.content) {
        if (!isRecord(block)) continue;
        if (block.type === 'text' && typeof block.text === 'string') {
          turn.parts.push({ type: 'text', text: block.text });
        } else if (block.type === 'image' && isRecord(block.source)) {
          if (block.source.type === 'base64' && typeof block.source.data === 'string') {
            turn.parts.push({
              type: 'image',
              mediaType: typeof block.source.media_type === 'string' ? block.source.media_type : 'image/png',
              data: block.source.data,
            });
          } else if (block.source.type === 'url' && typeof block.source.url === 'string') {
            const part = openAIImagePart(block.source.url);
            if (part !== null) turn.parts.push(part);
          }
        } else if (block.type === 'tool_use') {
          const name = typeof block.name === 'string' ? block.name : '';
          const id = typeof block.id === 'string' ? block.id : '';
          turn.toolCalls.push({ id, name, argumentsJson: JSON.stringify(isRecord(block.input) ? block.input : {}) });
          if (id !== '') toolNameById.set(id, name);
        } else if (block.type === 'tool_result') {
          turn.toolResults.push({
            id: typeof block.tool_use_id === 'string' ? block.tool_use_id : '',
            text: toolResultText(block.content),
          });
        }
      }
    }
    turns.push(turn);
  }

  const tools: ToolDef[] = [];
  if (Array.isArray(body.tools)) {
    for (const raw of body.tools) {
      if (!isRecord(raw) || typeof raw.name !== 'string') continue;
      tools.push({
        name: raw.name,
        description: typeof raw.description === 'string' ? raw.description : '',
        schema: isRecord(raw.input_schema) ? raw.input_schema : { type: 'object', properties: {} },
      });
    }
  }

  let toolChoice: string | null = null;
  if (isRecord(body.tool_choice)) {
    if (body.tool_choice.type === 'auto') toolChoice = 'auto';
    else if (body.tool_choice.type === 'any') toolChoice = 'required';
    else if (body.tool_choice.type === 'tool' && typeof body.tool_choice.name === 'string') toolChoice = body.tool_choice.name;
  }

  return {
    systemText: blocksText(body.system),
    turns,
    tools,
    toolChoice,
    maxTokens: numberOrNull(body.max_tokens),
    stops: stringList(body.stop_sequences),
    temperature: numberOrNull(body.temperature),
    jsonMode: false, // The Messages wire has no JSON-mode flag to echo.
    stream: body.stream === true,
    toolNameById,
  };
}

/** A normalized request as an OpenAI Chat Completions body naming `model`. */
export function toOpenAI(n: NormalizedRequest, model: string): Json {
  const messages: Json[] = [];
  if (n.systemText !== '') messages.push({ role: 'system', content: n.systemText });
  for (const turn of n.turns) {
    for (const result of turn.toolResults) {
      messages.push({ role: 'tool', tool_call_id: result.id, content: result.text });
    }
    if (turn.role === 'assistant') {
      const text = turn.parts.filter((p) => p.type === 'text').map((p) => (p.type === 'text' ? p.text : '')).join('\n');
      messages.push({
        role: 'assistant',
        content: text === '' ? null : text,
        ...(turn.toolCalls.length > 0
          ? { tool_calls: turn.toolCalls.map((c) => ({ id: c.id, type: 'function', function: { name: c.name, arguments: c.argumentsJson } })) }
          : {}),
      });
      continue;
    }
    if (turn.parts.length === 0) {
      // An empty user turn still resets what counts as the last user text; keep it visible.
      if (turn.toolResults.length > 0) messages.push({ role: 'user', content: '' });
      continue;
    }
    if (turn.parts.length === 1 && turn.parts[0].type === 'text') {
      messages.push({ role: 'user', content: turn.parts[0].text });
    } else {
      messages.push({
        role: 'user',
        content: turn.parts.map((p) =>
          p.type === 'text'
            ? { type: 'text', text: p.text }
            : { type: 'image_url', image_url: { url: p.type === 'image' ? dataUrlOf(p.mediaType, p.data) : p.url } },
        ),
      });
    }
  }
  const out: Json = { model, messages };
  if (n.maxTokens !== null) out.max_tokens = n.maxTokens;
  if (n.temperature !== null) out.temperature = n.temperature;
  if (n.stops.length > 0) out.stop = n.stops;
  if (n.tools.length > 0) {
    out.tools = n.tools.map((t) => ({ type: 'function', function: { name: t.name, description: t.description, parameters: t.schema } }));
  }
  if (n.toolChoice === 'auto' || n.toolChoice === 'required') out.tool_choice = n.toolChoice;
  else if (n.toolChoice !== null) out.tool_choice = { type: 'function', function: { name: n.toolChoice } };
  if (n.stream) {
    out.stream = true;
    // Ask for the usage chunk explicitly: the client checks the usage the provider reports,
    // and on this wire a stream only carries it when it was asked for.
    out.stream_options = { include_usage: true };
  }
  return out;
}

/** A tool call's arguments JSON back into an object; {} when it is not one. */
function argumentsObject(json: string): Json {
  try {
    const parsed: unknown = JSON.parse(json);
    if (isRecord(parsed)) return parsed;
  } catch {
    // Not JSON: an empty input.
  }
  return {};
}

/** A normalized request as an Anthropic Messages body naming `model`. */
export function toAnthropic(n: NormalizedRequest, model: string): Json {
  const messages: Json[] = [];
  for (const turn of n.turns) {
    const blocks: Json[] = [];
    for (const p of turn.parts) {
      if (p.type === 'text') blocks.push({ type: 'text', text: p.text });
      else if (p.type === 'image') blocks.push({ type: 'image', source: { type: 'base64', media_type: p.mediaType, data: p.data } });
      else blocks.push({ type: 'image', source: { type: 'url', url: p.url } });
    }
    for (const call of turn.toolCalls) {
      blocks.push({ type: 'tool_use', id: call.id, name: call.name, input: argumentsObject(call.argumentsJson) });
    }
    for (const result of turn.toolResults) {
      blocks.push({ type: 'tool_result', tool_use_id: result.id, content: result.text });
    }
    if (blocks.length === 0) blocks.push({ type: 'text', text: '' });
    messages.push({ role: turn.role, content: blocks });
  }
  const out: Json = { model, messages };
  if (n.systemText !== '') out.system = n.systemText;
  if (n.maxTokens !== null) out.max_tokens = n.maxTokens;
  if (n.temperature !== null) out.temperature = n.temperature;
  if (n.stops.length > 0) out.stop_sequences = n.stops;
  if (n.tools.length > 0) {
    out.tools = n.tools.map((t) => ({ name: t.name, description: t.description, input_schema: t.schema }));
  }
  if (n.toolChoice === 'auto') out.tool_choice = { type: 'auto' };
  else if (n.toolChoice === 'required') out.tool_choice = { type: 'any' };
  else if (n.toolChoice !== null) out.tool_choice = { type: 'tool', name: n.toolChoice };
  if (n.stream) out.stream = true;
  return out;
}

/** A normalized request as a Gemini generateContent body, with the path that names the model. */
export function toGemini(n: NormalizedRequest, model: string): { path: string; body: Json } {
  const contents: Json[] = [];
  for (const turn of n.turns) {
    const parts: Json[] = [];
    for (const p of turn.parts) {
      if (p.type === 'text') parts.push({ text: p.text });
      else if (p.type === 'image') parts.push({ inlineData: { mimeType: p.mediaType, data: p.data } });
      else parts.push({ fileData: { fileUri: p.url } });
    }
    for (const call of turn.toolCalls) {
      parts.push({ functionCall: { name: call.name, args: argumentsObject(call.argumentsJson) } });
    }
    for (const result of turn.toolResults) {
      parts.push({ functionResponse: { name: n.toolNameById.get(result.id) ?? '', response: { output: result.text } } });
    }
    if (parts.length === 0) parts.push({ text: '' });
    contents.push({ role: turn.role === 'assistant' ? 'model' : 'user', parts });
  }
  const body: Json = { contents };
  if (n.systemText !== '') body.systemInstruction = { parts: [{ text: n.systemText }] };
  if (n.tools.length > 0) {
    body.tools = [{ functionDeclarations: n.tools.map((t) => ({ name: t.name, description: t.description, parameters: t.schema })) }];
  }
  const config: Json = {};
  if (n.maxTokens !== null) config.maxOutputTokens = n.maxTokens;
  if (n.stops.length > 0) config.stopSequences = n.stops;
  if (n.temperature !== null) config.temperature = n.temperature;
  if (n.jsonMode) config.responseMimeType = 'application/json';
  if (Object.keys(config).length > 0) body.generationConfig = config;
  const path = `/v1beta/models/${model}:${n.stream ? 'streamGenerateContent' : 'generateContent'}`;
  return { path, body };
}

/** What a provider answered, in wire-free form: its whole text and its usage, in tokens. */
export interface ProviderAnswer {
  text: string;
  inputTokens: number;
  outputTokens: number;
}

function sseDataPayloads(body: string): string[] {
  const payloads: string[] = [];
  for (const frame of body.split('\n\n')) {
    for (const line of frame.split('\n')) {
      const trimmed = line.trim();
      if (trimmed.startsWith('data:')) payloads.push(trimmed.slice(5).trim());
    }
  }
  return payloads;
}

function parseJson(text: string): Json | null {
  try {
    const value: unknown = JSON.parse(text);
    return isRecord(value) ? value : null;
  } catch {
    return null;
  }
}

function usageOrZero(input: unknown, output: unknown): { inputTokens: number; outputTokens: number } {
  return {
    inputTokens: typeof input === 'number' ? input : 0,
    outputTokens: typeof output === 'number' ? output : 0,
  };
}

/** A provider's raw answer (JSON body or SSE stream, on its own wire) in wire-free form. */
export function parseProviderAnswer(wire: Wire, stream: boolean, bodyText: string): ProviderAnswer | null {
  if (wire === 'openai') {
    if (!stream) {
      const body = parseJson(bodyText);
      if (body === null) return null;
      const choices = Array.isArray(body.choices) ? body.choices : [];
      const choice = isRecord(choices[0]) ? choices[0] : {};
      const message = isRecord(choice.message) ? choice.message : {};
      const text = typeof message.content === 'string' ? message.content : blocksText(message.content);
      const usage = isRecord(body.usage) ? body.usage : {};
      return { text, ...usageOrZero(usage.prompt_tokens, usage.completion_tokens) };
    }
    let text = '';
    let usage = { inputTokens: 0, outputTokens: 0 };
    for (const payload of sseDataPayloads(bodyText)) {
      if (payload === '[DONE]') continue;
      const chunk = parseJson(payload);
      if (chunk === null) continue;
      if (isRecord(chunk.usage)) usage = usageOrZero(chunk.usage.prompt_tokens, chunk.usage.completion_tokens);
      const choices = Array.isArray(chunk.choices) ? chunk.choices : [];
      const choice = isRecord(choices[0]) ? choices[0] : null;
      if (choice !== null && isRecord(choice.delta) && typeof choice.delta.content === 'string') text += choice.delta.content;
    }
    return { text, ...usage };
  }
  if (wire === 'anthropic') {
    if (!stream) {
      const body = parseJson(bodyText);
      if (body === null) return null;
      const usage = isRecord(body.usage) ? body.usage : {};
      return { text: blocksText(body.content), ...usageOrZero(usage.input_tokens, usage.output_tokens) };
    }
    let text = '';
    let inputTokens = 0;
    let outputTokens = 0;
    for (const payload of sseDataPayloads(bodyText)) {
      const event = parseJson(payload);
      if (event === null) continue;
      if (event.type === 'message_start' && isRecord(event.message) && isRecord(event.message.usage)) {
        inputTokens = typeof event.message.usage.input_tokens === 'number' ? event.message.usage.input_tokens : 0;
      } else if (event.type === 'content_block_delta' && isRecord(event.delta) && typeof event.delta.text === 'string') {
        text += event.delta.text;
      } else if (event.type === 'message_delta' && isRecord(event.usage)) {
        outputTokens = typeof event.usage.output_tokens === 'number' ? event.usage.output_tokens : 0;
      }
    }
    return { text, inputTokens, outputTokens };
  }
  // Gemini: candidates carry the text in parts; usage rides usageMetadata, whole on a JSON
  // answer and on the stream's last frame.
  let text = '';
  let usage = { inputTokens: 0, outputTokens: 0 };
  const take = (body: Json): void => {
    const candidates = Array.isArray(body.candidates) ? body.candidates : [];
    const candidate = isRecord(candidates[0]) ? candidates[0] : {};
    const content = isRecord(candidate.content) ? candidate.content : {};
    if (Array.isArray(content.parts)) {
      for (const part of content.parts) {
        if (isRecord(part) && typeof part.text === 'string') text += part.text;
      }
    }
    if (isRecord(body.usageMetadata)) {
      usage = usageOrZero(body.usageMetadata.promptTokenCount, body.usageMetadata.candidatesTokenCount);
    }
  };
  if (!stream) {
    const body = parseJson(bodyText);
    if (body === null) return null;
    take(body);
  } else {
    for (const payload of sseDataPayloads(bodyText)) {
      const frame = parseJson(payload);
      if (frame !== null) take(frame);
    }
  }
  return { text, ...usage };
}

/** One server-sent event frame: an `event:` line when named, the data line, the blank line. */
export function sseEvent(event: string | null, data: unknown): string {
  return `${event === null ? '' : `event: ${event}\n`}data: ${JSON.stringify(data)}\n\n`;
}

/**
 * A provider answer rendered on the client's wire, in the client's streaming mode. The text is
 * the provider's, whole; the usage is the provider's, under the client's field names; the stop
 * reason is the natural end-of-turn word of the client's wire (the providers here always end
 * their turns naturally, so nothing else can arise).
 */
export function renderClientAnswer(api: ClientApi, stream: boolean, model: string, answer: ProviderAnswer): { contentType: string; body: string } {
  if (api === 'openai') {
    if (!stream) {
      return {
        contentType: 'application/json',
        body: JSON.stringify({
          id: 'chatcmpl-aggregator',
          object: 'chat.completion',
          created: FIXED_CREATED,
          model,
          choices: [{ index: 0, message: { role: 'assistant', content: answer.text }, finish_reason: 'stop' }],
          usage: {
            prompt_tokens: answer.inputTokens,
            completion_tokens: answer.outputTokens,
            total_tokens: answer.inputTokens + answer.outputTokens,
          },
        }),
      };
    }
    const chunk = (delta: Json, finishReason: string | null): Json => ({
      id: 'chatcmpl-aggregator',
      object: 'chat.completion.chunk',
      created: FIXED_CREATED,
      model,
      choices: [{ index: 0, delta, finish_reason: finishReason }],
    });
    let body = sseEvent(null, chunk({ role: 'assistant', content: '' }, null));
    if (answer.text !== '') body += sseEvent(null, chunk({ content: answer.text }, null));
    body += sseEvent(null, chunk({}, 'stop'));
    body += sseEvent(null, {
      id: 'chatcmpl-aggregator',
      object: 'chat.completion.chunk',
      created: FIXED_CREATED,
      model,
      choices: [],
      usage: { prompt_tokens: answer.inputTokens, completion_tokens: answer.outputTokens, total_tokens: answer.inputTokens + answer.outputTokens },
    });
    body += 'data: [DONE]\n\n';
    return { contentType: 'text/event-stream', body };
  }
  if (!stream) {
    return {
      contentType: 'application/json',
      body: JSON.stringify({
        id: 'msg_aggregator',
        type: 'message',
        role: 'assistant',
        model,
        content: [{ type: 'text', text: answer.text }],
        stop_reason: 'end_turn',
        stop_sequence: null,
        usage: { input_tokens: answer.inputTokens, output_tokens: answer.outputTokens },
      }),
    };
  }
  let body = sseEvent('message_start', {
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
  body += sseEvent('content_block_start', { type: 'content_block_start', index: 0, content_block: { type: 'text', text: '' } });
  if (answer.text !== '') {
    body += sseEvent('content_block_delta', { type: 'content_block_delta', index: 0, delta: { type: 'text_delta', text: answer.text } });
  }
  body += sseEvent('content_block_stop', { type: 'content_block_stop', index: 0 });
  body += sseEvent('message_delta', {
    type: 'message_delta',
    delta: { stop_reason: 'end_turn', stop_sequence: null },
    usage: { output_tokens: answer.outputTokens },
  });
  body += sseEvent('message_stop', { type: 'message_stop' });
  return { contentType: 'text/event-stream', body };
}

/** GET /v1/models in OpenAI's list shape: exactly the given free ids, no others. */
export function openAIModelList(ids: string[]): Json {
  return {
    object: 'list',
    data: ids.map((id) => ({ id, object: 'model', created: FIXED_CREATED, owned_by: 'aggregator', free: true })),
  };
}

/** GET /v1/models in Anthropic's list shape: exactly the given free ids, no others. */
export function anthropicModelList(ids: string[]): Json {
  return {
    data: ids.map((id) => ({ type: 'model', id, display_name: id, created_at: FIXED_CREATED_AT, free: true })),
    has_more: false,
    first_id: ids.length > 0 ? ids[0] : null,
    last_id: ids.length > 0 ? ids[ids.length - 1] : null,
  };
}

/** An error body in the client's wire shape. */
export function clientError(api: ClientApi, message: string): Json {
  return api === 'openai'
    ? { error: { message, type: 'api_error', code: null } }
    : { type: 'error', error: { type: 'api_error', message } };
}
