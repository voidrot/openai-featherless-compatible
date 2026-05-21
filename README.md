# @voidrot/openai-featherless-compatible

[![npm version](https://img.shields.io/npm/v/%40voidrot%2Fopenai-featherless-compatible)](https://www.npmjs.com/package/@voidrot/openai-featherless-compatible)
[![CI](https://github.com/voidrot/openai-featherless-compatible/actions/workflows/ci.yml/badge.svg)](https://github.com/voidrot/openai-featherless-compatible/actions/workflows/ci.yml)
[![Publish Package](https://github.com/voidrot/openai-featherless-compatible/actions/workflows/publish.yml/badge.svg)](https://github.com/voidrot/openai-featherless-compatible/actions/workflows/publish.yml)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](./LICENSE)

`@voidrot/openai-featherless-compatible` wraps `@ai-sdk/openai-compatible` with client-side fallback parsing for models that emit tool calls as text instead of native function-call objects.

This package is meant for OpenAI-compatible endpoints that work in OpenCode but occasionally return tool calls inside text blocks or XML-style envelopes. The fallback layer normalizes those responses into AI SDK `tool-call` events so OpenCode can execute tools normally.

## Why This Exists

Some OpenAI-compatible backends return tool calls as plain text instead of native function-call objects. That breaks downstream tool execution even when the model is otherwise usable.

This package restores that compatibility by:

- detecting text-emitted tool calls in common provider formats
- converting them into AI SDK tool-call objects and stream events
- cleaning tool-call envelopes and reasoning markers out of visible output
- coercing malformed tool arguments against function schemas when possible

## Installation

```bash
npm install @voidrot/openai-featherless-compatible
```

## Requirements

- Node.js 18+
- An OpenAI-compatible endpoint such as Featherless, DeepSeek, MiniMax, LM Studio, Ollama, or a compatible gateway
- An AI SDK consumer such as OpenCode or a direct AI SDK integration

## What It Covers

- Fenced JSON tool calls used by Qwen-style outputs
- Gemma 4 tool-call tags
- Kimi K2 tool-call sections
- DeepSeek V3 markers
- Mistral `[TOOL_CALLS]`
- MiniMax and DSML XML tool calls
- Generic XML tool-call formats such as `<tool_call>` and `<function name="...">`
- Schema-based coercion for malformed fallback tool arguments
- Streaming and non-streaming fallback paths with `tool-calls` finish reasons

## API Surface

The package exports:

- `createFeatherlessCompatibleProvider`
- `FeatherlessCompatibleChatLanguageModel`
- parser helpers such as `detectAndParseToolCalls` and `stripToolCallMarkers`
- fallback mode types such as `FeatherlessCompatibleToolCallFallbackMode`

Use `createFeatherlessCompatibleProvider` unless you specifically need the lower-level parser or model wrapper types.

## OpenCode Example

OpenCode custom providers load the npm package named in `provider.<id>.npm` and call the first named export that starts with `create`. This package exports `createFeatherlessCompatibleProvider`, so it works as a drop-in custom provider package.

Example `opencode.json`:

```json
{
  "$schema": "https://opencode.ai/config.json",
  "provider": {
    "featherless-ai": {
      "npm": "@voidrot/openai-featherless-compatible",
      "name": "Featherless AI",
      "options": {
        "baseURL": "https://api.featherless.ai/v1",
        "apiKey": "{env:FEATHERLESS_API_KEY}",
        "toolCallFallback": "auto"
      },
      "models": {
        ...
      }
    }
  }
}
```

Swap `baseURL`, `apiKey`, and model IDs for any OpenAI-compatible endpoint you want to use with the fallback parser, such as DeepSeek, MiniMax, LM Studio, Ollama, or another proxy/gateway.

`toolCallFallback` can be set to `auto`, `fenced-json`, `gemma4`, `kimi-k2`, `deepseek-v3`, `mistral`, `minimax`, `dsml`, `xml`, or `disabled`. `hermes` remains as a deprecated alias for `fenced-json`.

## Direct AI SDK Usage

```ts
import { createFeatherlessCompatibleProvider } from '@voidrot/openai-featherless-compatible';

const provider = createFeatherlessCompatibleProvider({
  name: 'featherless-ai',
  baseURL: 'https://api.featherless.ai/v1',
  apiKey: process.env.FEATHERLESS_API_KEY,
  toolCallFallback: 'auto',
});

const model = provider('kimi-k2');
```

## Development

```bash
npm install
npm run build
npm test
```

## CI And Releases

- Pull requests and pushes to `main` run the GitHub Actions CI workflow
- Version tags matching `v*` trigger the npm publish workflow
- Publishing uses npm trusted publishing with provenance enabled

## Contributing

Issues and pull requests are welcome.

If you are changing parser behavior or fallback mode semantics, include regression coverage for both direct generation and streaming paths.

## License

This project is released under the MIT license. See [LICENSE](./LICENSE).
