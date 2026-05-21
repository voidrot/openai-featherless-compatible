# @voidrot/openai-featherless-compatible

`@voidrot/openai-featherless-compatible` wraps `@ai-sdk/openai-compatible` with client-side fallback parsing for models that emit tool calls as text instead of native function-call objects.

This package is meant for OpenAI-compatible endpoints that work in OpenCode but occasionally return tool calls inside text blocks or XML-style envelopes. The fallback layer normalizes those responses into AI SDK `tool-call` events so OpenCode can execute tools normally.

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
