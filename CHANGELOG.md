# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.2.0] - 2026-05-24

### Added

- Support for StepFun-style `<function=name>` / `<parameter=key>` tool-call syntax, used by models such as `stepfun-ai/Step-3.5-Flash`, in both detection, parsing, and stripping paths
- Fallback-emitted stream reasoning parts now use unique IDs (`reasoning-fallback-N`) to prevent collisions when upstream already emits reasoning events in the same stream

### Fixed

- Strip tool-call markers from within reasoning content to preserve clean reasoning text
- Properly inject reasoning content into response even when no actual tool calls are detected
- Fixed whitespace normalization after marker stripping to prevent double spaces
- Fixed duplicate extraction bug when `<function>` tags appear inside `<tool_call>` wrapper blocks — wrapper contents are now parsed recursively and top-level function tags are only matched outside wrappers

### Tests

- Added regression coverage for StepFun `<function=name>/<parameter=key>` format in parser and end-to-end generate paths
- Added regression tests for message integrity invariants: stream chunk boundaries and IDs are preserved when no fallback should apply
- Added regression tests for multi-tool fallback parsing in both `doGenerate` and `doStream` paths
- Added regression tests for disabled-mode and forced-format-mismatch no-op guarantees
- Added regression tests for native tool-call precedence over fallback parsing
- Added regression tests for non-text part interleaving, stream flush without finish, non-parsable marker preservation, non-text event ordering, reasoning ID uniqueness, and schema coercion with multi-tool streaming

## [0.1.3] - 2026-05-24

### Fixed

- Handle `<function=name>` alternative tool-call marker syntax in reasoning blocks
- Strip tool-call markers from within reasoning content to preserve clean reasoning text
- Properly inject reasoning content into response even when no actual tool calls are detected
- Fixed whitespace normalization after marker stripping to prevent double spaces

### Improvements

- Enhanced `stripXml()` function to recognize and remove both `<function name="">` and `<function=name>` formats
- Improved reasoning content extraction to handle edge cases with embedded tool markers
- Better separation of reasoning and tool-call content in fallback parsing

## [0.1.2] - 2026-05-10

### Streaming Support

- Streaming support for reasoning lifecycle events (`reasoning-start`, `reasoning-delta`, `reasoning-end`)
- Emit reasoning content as separate events in streaming responses
- Proper lifecycle management for reasoning blocks in streamed output

### Enhancements

- Enhanced `stream.ts` to emit reasoning events before text events
- Better integration between detection results and streaming response handling

## [0.1.1] - 2026-04-26

### Reasoning Features

- Reasoning content extraction from XML tags (`<think>`, `<thinking>`, `<reasoning>`, `<thought>`, `<REASONING_SCRATCHPAD>`)
- Support for both closed `<tag>content</tag>` and unclosed trailing reasoning tags
- Reasoning injection into AI SDK responses as `reasoning` content parts
- Tool-call marker stripping from extracted reasoning content
- Conditional reasoning part injection when not already present in response

### API Changes

- Modified `detectAndParseToolCalls()` to extract and return reasoning content separately
- Updated `model.ts` to inject reasoning parts into fallback response paths
- Enhanced `ToolCallDetection` interface with optional `reasoningContent` field
- Multi-turn context preservation by retaining reasoning blocks during fallback parsing
- Better content separation between reasoning and cleaned text output

## [0.1.0] - 2026-04-12

### Initial Release

- Core fallback tool-call detection from text content
- Support for multiple tool-call marker formats:
  - `<function name="">...</function>` (standard)
  - `<tool_call>...</tool_call>` (wrapper)
  - `<tool_calls>...</tool_calls>` (wrapper)
  - `<function_call>...</function_call>` (wrapper)
  - `<function_calls>...</function_calls>` (wrapper)
- Format-specific parsing with registered handlers for xml, markdown, and text formats
- Fallback provider integration with OpenAI-compatible models
- Tool call extraction, validation, and reindexing
- Comprehensive test suite with 52+ test cases
- Automatic detection when provider returns text instead of structured tool calls
- Graceful fallback when native tool calling is unavailable
- Support for both generated and streamed responses
- XML/HTML tag stripping with format-specific handling

[0.2.0]: https://github.com/voidrot/openai-featherless-compatible/compare/0.1.3...0.2.0
[0.1.3]: https://github.com/voidrot/openai-featherless-compatible/compare/0.1.2...0.1.3
[0.1.2]: https://github.com/voidrot/openai-featherless-compatible/compare/0.1.1...0.1.2
[0.1.1]: https://github.com/voidrot/openai-featherless-compatible/compare/0.1.0...0.1.1
[0.1.0]: https://github.com/voidrot/openai-featherless-compatible/releases/tag/0.1.0
