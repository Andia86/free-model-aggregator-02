# Free Model Aggregator — Live

[![make.fun · free-model-aggregator-02](https://make.fun/badge/free-model-aggregator-02.svg)](https://make.fun/bounty/free-model-aggregator-02?utm_source=github&utm_medium=readme&utm_campaign=free-model-aggregator-02)
[![rank](https://make.fun/badge/free-model-aggregator-02/Andia86/free-model-aggregator-02/rank.svg)](https://make.fun/bounty/free-model-aggregator-02/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=free-model-aggregator-02)
[![reward](https://make.fun/badge/free-model-aggregator-02/Andia86/free-model-aggregator-02/reward.svg)](https://make.fun/bounty/free-model-aggregator-02?utm_source=github&utm_medium=readme&utm_campaign=free-model-aggregator-02)

A routing server for make.fun's **free-model-aggregator-02** ("The best live free-model
aggregator"): it aggregates the live catalog of OpenRouter free models behind one endpoint
and answers every request with the best free model that can actually carry it. Every
completion is a real call to an OpenRouter free model, dialed through the gateway on the
key pushed to the competition.

The routing core is the one that scored 99 on all three practice trials of
free-model-aggregator-01, adapted to the live brief: the brief's schema names are matched
by document kind and version rather than by competition slug, and the provider key comes
from the brief's own key or the pushed secret it names.

## What it does

The trial brief hands the server a live catalog of OpenRouter's free models, the house's
quality ratings for them, and a workload of requests (OpenAI and Anthropic wires, JSON and
streaming, some with a system prompt, one declaring tools, some in JSON mode). This server:

1. **Builds its catalog from the brief alone** — the provider directory plus the quality
   table are everything the registered document needs, so the catalog is complete and
   identical between `node main.ts --list` and the register body.
2. **Plans per request.** Candidates are the best free models that meet the request's
   requirements (tools, vision, JSON mode, minimum context) on the quality table's word.
   A request the client put in JSON mode only rides a wire that can echo JSON mode back
   (OpenAI or Gemini) — the Anthropic wire has no such flag, so it sits those out.
   Paid models are never candidates.
3. **Translates across wires.** Clients speak OpenAI or Anthropic; providers also speak
   Gemini. Same-wire traffic passes through untouched. Cross-wire traffic goes through a
   normalized form (`lib/wire.ts`) that preserves exactly what a provider echoes about a
   request — system text, last user text, tool names in order, output cap, stop sequences,
   image bytes, JSON mode — and renders the answer back with the provider's own text,
   usage and stop semantics. Streamed OpenAI requests explicitly ask for the usage chunk,
   because a stream only carries usage when it was asked for.
4. **Fails over, rating first.** A refused call (a real 429 or 5xx, budget spent, outage)
   moves to the next candidate, but rating always leads the order: a refusal only costs a
   provider its place against candidates rated within 15 milli of it, and it is forgotten
   once two further requests have been handled (or at once, when the provider answers
   again), so one refusal cannot bench the best model for the whole trial. Equally rated
   candidates are taken from the provider called least so far. One request gives up after
   six refused calls. Exactly one provider call may succeed per request, and its answer is
   the one served.
5. **Is deterministic.** Every choice derives from the brief, the request and the
   providers' answers: no clock, no randomness.

## Layout

- `main.ts` — the server, the planner and the trial lifecycle (listen, register, poll
  `/work`, exit 0 when the work is done).
- `lib/brief.ts` — reads the three JSON documents out of the brief; builds the catalog.
- `lib/wire.ts` — the normalized request form and the OpenAI / Anthropic / Gemini
  translations, both directions, streamed and not.
- `site/index.html` — a standalone showcase page for the aggregator.

## Running it

The entry is plain TypeScript on Node's standard library only:

```sh
node main.ts --list   # print the catalog document the server registers
node main.ts          # inside a trial: serve, register, work, exit
```

MIT licensed — see `LICENSE`.
