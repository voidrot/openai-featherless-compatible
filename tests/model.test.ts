import { describe, expect, it, vi } from "vitest";
import type {
  LanguageModelV3FinishReason,
  LanguageModelV3GenerateResult,
  LanguageModelV3StreamPart,
  LanguageModelV3StreamResult,
  LanguageModelV3Usage,
} from "@ai-sdk/provider";
import { FeatherlessCompatibleChatLanguageModel } from "../src/model";

const usage: LanguageModelV3Usage = {
  inputTokens: {
    total: 10,
    noCache: 10,
    cacheRead: 0,
    cacheWrite: 0,
  },
  outputTokens: {
    total: 4,
    text: 4,
    reasoning: 0,
  },
  raw: undefined,
};

function createFinishReason(
  unified: LanguageModelV3FinishReason["unified"],
): LanguageModelV3FinishReason {
  return { unified, raw: unified === "tool-calls" ? "tool_calls" : unified };
}

function createStream(
  parts: LanguageModelV3StreamPart[],
): ReadableStream<LanguageModelV3StreamPart> {
  return new ReadableStream<LanguageModelV3StreamPart>({
    start(controller) {
      for (const part of parts) {
        controller.enqueue(part);
      }
      controller.close();
    },
  });
}

async function readStreamParts(
  stream: ReadableStream<LanguageModelV3StreamPart>,
): Promise<LanguageModelV3StreamPart[]> {
  const reader = stream.getReader();
  const parts: LanguageModelV3StreamPart[] = [];

  while (true) {
    const { done, value } = await reader.read();
    if (done) {
      return parts;
    }

    parts.push(value);
  }
}

function createModel(params: {
  generateResult?: LanguageModelV3GenerateResult;
  streamParts?: LanguageModelV3StreamPart[];
  fallback?: "auto" | "fenced-json" | "gemma4" | "disabled";
}) {
  const inner = {
    specificationVersion: "v3" as const,
    provider: "test.chat",
    modelId: "test-model",
    supportedUrls: {},
    doGenerate: vi.fn(async () => params.generateResult),
    doStream: vi.fn(
      async (): Promise<LanguageModelV3StreamResult> => ({
        stream: createStream(params.streamParts ?? []),
        request: { body: {} },
        response: { headers: new Headers() },
      }),
    ),
  };

  const baseProvider = {
    name: "test-provider",
    languageModel: vi.fn(() => inner),
  };

  return new FeatherlessCompatibleChatLanguageModel(
    baseProvider as never,
    "test-model",
    params.fallback ?? "auto",
  );
}

describe("FeatherlessCompatibleChatLanguageModel", () => {
  it("marks fallback-generated tool calls as tool-calls in doGenerate", async () => {
    const model = createModel({
      generateResult: {
        content: [
          {
            type: "text",
            text: 'I will call a tool.\n```json\n{"name":"search","arguments":{"query":"opencode"}}\n```',
          },
        ],
        finishReason: createFinishReason("stop"),
        usage,
        warnings: [],
      },
    });

    const result = await model.doGenerate({
      prompt: [],
      headers: undefined,
      abortSignal: undefined,
    });

    expect(result.finishReason.unified).toBe("tool-calls");
    expect(result.content).toContainEqual({
      type: "tool-call",
      toolCallId: "tc_0_search",
      toolName: "search",
      input: '{"query":"opencode"}',
    });
    expect(result.content.find((part) => part.type === "text")).toMatchObject({
      type: "text",
      text: "I will call a tool.\n",
    });
  });

  it("coerces parsed fallback arguments using the tool schema", async () => {
    const model = createModel({
      generateResult: {
        content: [
          {
            type: "text",
            text: '<tool_call>{"name":"search","arguments":{"tags":"opencode","limit":"2","exact":"true"}}</tool_call>',
          },
        ],
        finishReason: createFinishReason("stop"),
        usage,
        warnings: [],
      },
    });

    const result = await model.doGenerate({
      prompt: [],
      headers: undefined,
      abortSignal: undefined,
      tools: [
        {
          type: "function",
          name: "search",
          inputSchema: {
            type: "object",
            properties: {
              tags: { type: "array", items: { type: "string" } },
              limit: { type: "integer" },
              exact: { type: "boolean" },
            },
          },
        },
      ],
    });

    expect(result.content).toContainEqual({
      type: "tool-call",
      toolCallId: "tc_0_search",
      toolName: "search",
      input: '{"tags":["opencode"],"limit":2,"exact":true}',
    });
  });

  it("converts streamed fallback text into tool call events and updates finish reason", async () => {
    const model = createModel({
      fallback: "fenced-json",
      streamParts: [
        { type: "stream-start", warnings: [] },
        {
          type: "response-metadata",
          id: "resp-1",
          modelId: "test-model",
          timestamp: new Date(0),
        },
        { type: "text-start", id: "txt-0" },
        {
          type: "text-delta",
          id: "txt-0",
          delta:
            'I will call a tool.\n```json\n{"name":"search","arguments":{"query":"opencode"}}\n```',
        },
        { type: "text-end", id: "txt-0" },
        { type: "finish", finishReason: createFinishReason("stop"), usage },
      ],
    });

    const result = await model.doStream({
      prompt: [],
      headers: undefined,
      abortSignal: undefined,
    });
    const parts = await readStreamParts(result.stream);

    expect(parts).toContainEqual({ type: "text-start", id: "txt-0" });
    expect(parts).toContainEqual({
      type: "text-delta",
      id: "txt-0",
      delta: "I will call a tool.\n",
    });
    expect(parts).toContainEqual({
      type: "tool-input-start",
      id: "tc_0_search",
      toolName: "search",
    });
    expect(parts).toContainEqual({
      type: "tool-call",
      toolCallId: "tc_0_search",
      toolName: "search",
      input: '{"query":"opencode"}',
    });

    const finish = parts.find((part) => part.type === "finish");
    expect(finish).toMatchObject({
      type: "finish",
      finishReason: { unified: "tool-calls" },
    });

    const streamedText = parts
      .filter(
        (
          part,
        ): part is Extract<LanguageModelV3StreamPart, { type: "text-delta" }> =>
          part.type === "text-delta",
      )
      .map((part) => part.delta)
      .join("");
    expect(streamedText).not.toContain("```json");
  });

  it("passes native streamed tool calls through unchanged", async () => {
    const streamParts: LanguageModelV3StreamPart[] = [
      { type: "stream-start", warnings: [] },
      { type: "text-start", id: "txt-0" },
      { type: "text-delta", id: "txt-0", delta: "before tool call" },
      { type: "text-end", id: "txt-0" },
      { type: "tool-input-start", id: "call-1", toolName: "search" },
      { type: "tool-input-delta", id: "call-1", delta: '{"query":"opencode"}' },
      { type: "tool-input-end", id: "call-1" },
      {
        type: "tool-call",
        toolCallId: "call-1",
        toolName: "search",
        input: '{"query":"opencode"}',
      },
      { type: "finish", finishReason: createFinishReason("tool-calls"), usage },
    ];
    const model = createModel({ streamParts });

    const result = await model.doStream({
      prompt: [],
      headers: undefined,
      abortSignal: undefined,
    });
    const parts = await readStreamParts(result.stream);

    expect(parts).toEqual(streamParts);
  });

  it("strips reasoning tags before emitting fallback text content", async () => {
    const model = createModel({
      generateResult: {
        content: [
          {
            type: "text",
            text: '<think>hidden</think>Visible <tool_call>{"name":"search","arguments":{}}</tool_call>',
          },
        ],
        finishReason: createFinishReason("stop"),
        usage,
        warnings: [],
      },
    });

    const result = await model.doGenerate({
      prompt: [],
      headers: undefined,
      abortSignal: undefined,
    });

    expect(
      result.content.find((part) => part.type === "reasoning"),
    ).toMatchObject({
      type: "reasoning",
      text: "hidden",
    });
    expect(result.content.find((part) => part.type === "text")).toMatchObject({
      type: "text",
      text: "Visible ",
    });
  });

  it("emits reasoning stream parts when fallback text contains think tags", async () => {
    const model = createModel({
      streamParts: [
        { type: "stream-start", warnings: [] },
        { type: "text-start", id: "txt-0" },
        {
          type: "text-delta",
          id: "txt-0",
          delta:
            '<think>step-by-step</think>Visible <tool_call>{"name":"search","arguments":{}}</tool_call>',
        },
        { type: "text-end", id: "txt-0" },
        { type: "finish", finishReason: createFinishReason("stop"), usage },
      ],
    });

    const result = await model.doStream({
      prompt: [],
      headers: undefined,
      abortSignal: undefined,
    });
    const parts = await readStreamParts(result.stream);

    const reasoningStart = parts.find(
      (part): part is Extract<LanguageModelV3StreamPart, { type: "reasoning-start" }> =>
        part.type === "reasoning-start",
    );
    expect(reasoningStart).toBeDefined();

    expect(parts).toContainEqual({
      type: "reasoning-delta",
      id: reasoningStart!.id,
      delta: "step-by-step",
    });
    expect(parts).toContainEqual({
      type: "reasoning-end",
      id: reasoningStart!.id,
    });
    expect(parts).toContainEqual({
      type: "text-delta",
      id: "txt-0",
      delta: "Visible ",
    });
  });

  it("strips tool-call markers from reasoning content in fallback", async () => {
    const model = createModel({
      generateResult: {
        content: [
          {
            type: "text",
            text: "<think>Calling <function=search></function> to lookup info. After that I think of answer</think>Result",
          },
        ],
        usage: { promptTokens: 10, completionTokens: 20 },
      },
    });

    const result = await model.doGenerate({
      prompt: [],
      headers: undefined,
      abortSignal: undefined,
    });

    expect(
      result.content.find((part) => part.type === "reasoning"),
    ).toMatchObject({
      type: "reasoning",
      text: "Calling to lookup info. After that I think of answer",
    });
    expect(result.content.find((part) => part.type === "text")).toMatchObject({
      type: "text",
      text: "Result",
    });
  });

  it("preserves multi-part assistant text when injecting reasoning without tool calls", async () => {
    const model = createModel({
      generateResult: {
        content: [
          { type: "text", text: "First assistant message." },
          {
            type: "text",
            text: "<think>hidden chain of thought</think>Second assistant message.",
          },
        ],
        finishReason: createFinishReason("stop"),
        usage,
        warnings: [],
      },
    });

    const result = await model.doGenerate({
      prompt: [],
      headers: undefined,
      abortSignal: undefined,
    });

    expect(result.content).toContainEqual({
      type: "text",
      text: "First assistant message.",
    });
    expect(result.content).toContainEqual({
      type: "text",
      text: "Second assistant message.",
    });
    expect(
      result.content.find((part) => part.type === "reasoning"),
    ).toMatchObject({
      type: "reasoning",
      text: "hidden chain of thought",
    });
  });

  it("parses fallback tool calls from later text parts without dropping earlier assistant text", async () => {
    const model = createModel({
      generateResult: {
        content: [
          { type: "text", text: "Intro assistant message." },
          {
            type: "text",
            text: '<tool_call>{"name":"search","arguments":{"query":"opencode"}}</tool_call>Final assistant message.',
          },
        ],
        finishReason: createFinishReason("stop"),
        usage,
        warnings: [],
      },
    });

    const result = await model.doGenerate({
      prompt: [],
      headers: undefined,
      abortSignal: undefined,
    });

    expect(result.finishReason.unified).toBe("tool-calls");
    expect(result.content).toContainEqual({
      type: "text",
      text: "Intro assistant message.",
    });
    expect(result.content).toContainEqual({
      type: "text",
      text: "Final assistant message.",
    });
    expect(result.content).toContainEqual({
      type: "tool-call",
      toolCallId: "tc_0_search",
      toolName: "search",
      input: '{"query":"opencode"}',
    });
  });

  it("preserves streamed text chunk boundaries and ids when no fallback markers are present", async () => {
    const streamParts: LanguageModelV3StreamPart[] = [
      { type: "stream-start", warnings: [] },
      { type: "text-start", id: "txt-0" },
      { type: "text-delta", id: "txt-0", delta: "alpha " },
      { type: "text-delta", id: "txt-0", delta: "beta" },
      { type: "text-end", id: "txt-0" },
      { type: "text-start", id: "txt-1" },
      { type: "text-delta", id: "txt-1", delta: " gamma" },
      { type: "text-end", id: "txt-1" },
      { type: "finish", finishReason: createFinishReason("stop"), usage },
    ];
    const model = createModel({ streamParts });

    const result = await model.doStream({
      prompt: [],
      headers: undefined,
      abortSignal: undefined,
    });
    const parts = await readStreamParts(result.stream);

    expect(parts).toEqual(streamParts);
  });

  it("supports multi tool calling in streamed fallback parsing", async () => {
    const model = createModel({
      streamParts: [
        { type: "stream-start", warnings: [] },
        { type: "text-start", id: "txt-0" },
        {
          type: "text-delta",
          id: "txt-0",
          delta:
            'Before <tool_call>{"name":"search","arguments":{"query":"one"}}</tool_call> and <tool_call>{"name":"get_weather","arguments":{"location":"NYC"}}</tool_call> after',
        },
        { type: "text-end", id: "txt-0" },
        { type: "finish", finishReason: createFinishReason("stop"), usage },
      ],
    });

    const result = await model.doStream({
      prompt: [],
      headers: undefined,
      abortSignal: undefined,
    });
    const parts = await readStreamParts(result.stream);

    const toolCalls = parts.filter(
      (part): part is Extract<LanguageModelV3StreamPart, { type: "tool-call" }> =>
        part.type === "tool-call",
    );

    expect(toolCalls).toEqual([
      {
        type: "tool-call",
        toolCallId: "tc_0_search",
        toolName: "search",
        input: '{"query":"one"}',
      },
      {
        type: "tool-call",
        toolCallId: "tc_1_get_weather",
        toolName: "get_weather",
        input: '{"location":"NYC"}',
      },
    ]);

    const text = parts
      .filter(
        (
          part,
        ): part is Extract<LanguageModelV3StreamPart, { type: "text-delta" }> =>
          part.type === "text-delta",
      )
      .map((part) => part.delta)
      .join("");

    expect(text).toContain("Before");
    expect(text).toContain("after");

    const finish = parts.find((part) => part.type === "finish");
    expect(finish).toMatchObject({
      type: "finish",
      finishReason: { unified: "tool-calls" },
    });
  });

  it("supports multi tool calling in doGenerate fallback across text parts", async () => {
    const model = createModel({
      generateResult: {
        content: [
          {
            type: "text",
            text: 'Before <tool_call>{"name":"search","arguments":{"query":"one"}}</tool_call>',
          },
          {
            type: "text",
            text: 'Middle <tool_call>{"name":"get_weather","arguments":{"location":"NYC"}}</tool_call> After',
          },
        ],
        finishReason: createFinishReason("stop"),
        usage,
        warnings: [],
      },
    });

    const result = await model.doGenerate({
      prompt: [],
      headers: undefined,
      abortSignal: undefined,
    });

    expect(result.finishReason.unified).toBe("tool-calls");
    expect(result.content).toContainEqual({
      type: "tool-call",
      toolCallId: "tc_0_search",
      toolName: "search",
      input: '{"query":"one"}',
    });
    expect(result.content).toContainEqual({
      type: "tool-call",
      toolCallId: "tc_1_get_weather",
      toolName: "get_weather",
      input: '{"location":"NYC"}',
    });
    expect(result.content).toContainEqual({
      type: "text",
      text: "Before ",
    });
    expect(result.content).toContainEqual({
      type: "text",
      text: "Middle After",
    });
  });

  it("is a no-op in doGenerate when fallback is disabled even with markers present", async () => {
    const generateResult: LanguageModelV3GenerateResult = {
      content: [
        {
          type: "text",
          text: '<think>hidden</think><tool_call>{"name":"search","arguments":{"query":"x"}}</tool_call>Visible',
        },
      ],
      finishReason: createFinishReason("stop"),
      usage,
      warnings: [],
    };
    const model = createModel({
      generateResult,
      fallback: "disabled",
    });

    const result = await model.doGenerate({
      prompt: [],
      headers: undefined,
      abortSignal: undefined,
    });

    expect(result).toEqual(generateResult);
  });

  it("is a no-op in doStream when fallback is disabled even with markers present", async () => {
    const streamParts: LanguageModelV3StreamPart[] = [
      { type: "stream-start", warnings: [] },
      { type: "text-start", id: "txt-0" },
      {
        type: "text-delta",
        id: "txt-0",
        delta:
          '<think>hidden</think><tool_call>{"name":"search","arguments":{"query":"x"}}</tool_call>Visible',
      },
      { type: "text-end", id: "txt-0" },
      { type: "finish", finishReason: createFinishReason("stop"), usage },
    ];
    const model = createModel({
      streamParts,
      fallback: "disabled",
    });

    const result = await model.doStream({
      prompt: [],
      headers: undefined,
      abortSignal: undefined,
    });
    const parts = await readStreamParts(result.stream);

    expect(parts).toEqual(streamParts);
  });

  it("prefers native streaming tool calls and does not emit fallback tool calls", async () => {
    const streamParts: LanguageModelV3StreamPart[] = [
      { type: "stream-start", warnings: [] },
      { type: "text-start", id: "txt-0" },
      {
        type: "text-delta",
        id: "txt-0",
        delta:
          'marker-like text <tool_call>{"name":"search","arguments":{"query":"x"}}</tool_call> that should stay raw',
      },
      { type: "text-end", id: "txt-0" },
      { type: "tool-input-start", id: "native-1", toolName: "search" },
      {
        type: "tool-input-delta",
        id: "native-1",
        delta: '{"query":"native"}',
      },
      { type: "tool-input-end", id: "native-1" },
      {
        type: "tool-call",
        toolCallId: "native-1",
        toolName: "search",
        input: '{"query":"native"}',
      },
      { type: "finish", finishReason: createFinishReason("tool-calls"), usage },
    ];
    const model = createModel({ streamParts });

    const result = await model.doStream({
      prompt: [],
      headers: undefined,
      abortSignal: undefined,
    });
    const parts = await readStreamParts(result.stream);

    expect(parts).toEqual(streamParts);
  });

  it("preserves non-text interleaving in doGenerate while applying fallback transforms", async () => {
    const model = createModel({
      generateResult: {
        content: [
          { type: "reasoning", text: "preexisting reasoning" },
          {
            type: "text",
            text: 'Before <tool_call>{"name":"search","arguments":{"query":"one"}}</tool_call>',
          },
          {
            type: "text",
            text: 'After <tool_call>{"name":"get_weather","arguments":{"location":"NYC"}}</tool_call>',
          },
        ],
        finishReason: createFinishReason("stop"),
        usage,
        warnings: [],
      },
    });

    const result = await model.doGenerate({
      prompt: [],
      headers: undefined,
      abortSignal: undefined,
    });

    expect(result.content[0]).toEqual({
      type: "reasoning",
      text: "preexisting reasoning",
    });
    expect(result.content[1]).toEqual({ type: "text", text: "Before " });
    expect(result.content[2]).toEqual({
      type: "tool-call",
      toolCallId: "tc_0_search",
      toolName: "search",
      input: '{"query":"one"}',
    });
    expect(result.content[3]).toEqual({ type: "text", text: "After " });
    expect(result.content[4]).toEqual({
      type: "tool-call",
      toolCallId: "tc_1_get_weather",
      toolName: "get_weather",
      input: '{"location":"NYC"}',
    });
  });

  it("uses unique reasoning ids in stream fallback when reasoning ids already exist", async () => {
    const model = createModel({
      streamParts: [
        { type: "stream-start", warnings: [] },
        { type: "reasoning-start", id: "reasoning-0" },
        { type: "reasoning-delta", id: "reasoning-0", delta: "existing" },
        { type: "reasoning-end", id: "reasoning-0" },
        { type: "text-start", id: "txt-0" },
        {
          type: "text-delta",
          id: "txt-0",
          delta:
            '<think>new fallback reasoning</think><tool_call>{"name":"search","arguments":{"query":"x"}}</tool_call>',
        },
        { type: "text-end", id: "txt-0" },
        { type: "finish", finishReason: createFinishReason("stop"), usage },
      ],
    });

    const result = await model.doStream({
      prompt: [],
      headers: undefined,
      abortSignal: undefined,
    });
    const parts = await readStreamParts(result.stream);

    const reasoningStarts = parts.filter(
      (part): part is Extract<LanguageModelV3StreamPart, { type: "reasoning-start" }> =>
        part.type === "reasoning-start",
    );

    const ids = reasoningStarts.map((part) => part.id);
    expect(new Set(ids).size).toBe(ids.length);
  });

  it("parses marker content split across text-delta chunks", async () => {
    const model = createModel({
      streamParts: [
        { type: "stream-start", warnings: [] },
        { type: "text-start", id: "txt-0" },
        { type: "text-delta", id: "txt-0", delta: "Before <tool_" },
        {
          type: "text-delta",
          id: "txt-0",
          delta: 'call>{"name":"search","arguments":{"query":"split"}}</tool_call> After',
        },
        { type: "text-end", id: "txt-0" },
        { type: "finish", finishReason: createFinishReason("stop"), usage },
      ],
    });

    const result = await model.doStream({
      prompt: [],
      headers: undefined,
      abortSignal: undefined,
    });
    const parts = await readStreamParts(result.stream);

    expect(parts).toContainEqual({
      type: "tool-call",
      toolCallId: "tc_0_search",
      toolName: "search",
      input: '{"query":"split"}',
    });
  });

  it("parses marker content split across multiple text segments", async () => {
    const model = createModel({
      streamParts: [
        { type: "stream-start", warnings: [] },
        { type: "text-start", id: "txt-0" },
        { type: "text-delta", id: "txt-0", delta: "Before <tool_" },
        { type: "text-end", id: "txt-0" },
        { type: "text-start", id: "txt-1" },
        {
          type: "text-delta",
          id: "txt-1",
          delta: 'call>{"name":"search","arguments":{"query":"segmented"}}</tool_call> After',
        },
        { type: "text-end", id: "txt-1" },
        { type: "finish", finishReason: createFinishReason("stop"), usage },
      ],
    });

    const result = await model.doStream({
      prompt: [],
      headers: undefined,
      abortSignal: undefined,
    });
    const parts = await readStreamParts(result.stream);

    expect(parts).toContainEqual({
      type: "tool-call",
      toolCallId: "tc_0_search",
      toolName: "search",
      input: '{"query":"segmented"}',
    });
  });

  it("coerces multi-tool arguments in streamed fallback parsing using tool schemas", async () => {
    const model = createModel({
      streamParts: [
        { type: "stream-start", warnings: [] },
        { type: "text-start", id: "txt-0" },
        {
          type: "text-delta",
          id: "txt-0",
          delta:
            '<tool_call>{"name":"search","arguments":{"limit":"2","exact":"true"}}</tool_call><tool_call>{"name":"set_flags","arguments":{"enabled":"false","retries":"3"}}</tool_call>',
        },
        { type: "text-end", id: "txt-0" },
        { type: "finish", finishReason: createFinishReason("stop"), usage },
      ],
    });

    const result = await model.doStream({
      prompt: [],
      headers: undefined,
      abortSignal: undefined,
      tools: [
        {
          type: "function",
          name: "search",
          inputSchema: {
            type: "object",
            properties: {
              limit: { type: "integer" },
              exact: { type: "boolean" },
            },
          },
        },
        {
          type: "function",
          name: "set_flags",
          inputSchema: {
            type: "object",
            properties: {
              enabled: { type: "boolean" },
              retries: { type: "integer" },
            },
          },
        },
      ],
    });
    const parts = await readStreamParts(result.stream);

    expect(parts).toContainEqual({
      type: "tool-call",
      toolCallId: "tc_0_search",
      toolName: "search",
      input: '{"limit":2,"exact":true}',
    });
    expect(parts).toContainEqual({
      type: "tool-call",
      toolCallId: "tc_1_set_flags",
      toolName: "set_flags",
      input: '{"enabled":false,"retries":3}',
    });
  });

  it("is a strict no-op in doGenerate when native tool-call parts already exist", async () => {
    const generateResult: LanguageModelV3GenerateResult = {
      content: [
        {
          type: "text",
          text: 'raw marker-looking text <tool_call>{"name":"search","arguments":{"query":"x"}}</tool_call>',
        },
        {
          type: "tool-call",
          toolCallId: "native-1",
          toolName: "search",
          input: '{"query":"native"}',
        },
      ],
      finishReason: createFinishReason("tool-calls"),
      usage,
      warnings: [],
    };
    const model = createModel({ generateResult });

    const result = await model.doGenerate({
      prompt: [],
      headers: undefined,
      abortSignal: undefined,
    });

    expect(result).toEqual(generateResult);
  });

  it("does not inject duplicate reasoning when reasoning parts already exist in doGenerate", async () => {
    const model = createModel({
      generateResult: {
        content: [
          { type: "reasoning", text: "existing reasoning" },
          {
            type: "text",
            text: "<think>fallback reasoning</think>Visible response",
          },
        ],
        finishReason: createFinishReason("stop"),
        usage,
        warnings: [],
      },
    });

    const result = await model.doGenerate({
      prompt: [],
      headers: undefined,
      abortSignal: undefined,
    });

    const reasoningParts = result.content.filter(
      (part): part is Extract<(typeof result.content)[number], { type: "reasoning" }> =>
        part.type === "reasoning",
    );

    expect(reasoningParts).toHaveLength(1);
    expect(reasoningParts[0]).toEqual({
      type: "reasoning",
      text: "existing reasoning",
    });
    expect(result.content).toContainEqual({
      type: "text",
      text: "Visible response",
    });
  });

  it("is a no-op when forced fallback format does not match content in doGenerate", async () => {
    const generateResult: LanguageModelV3GenerateResult = {
      content: [
        {
          type: "text",
          text: '<tool_call>{"name":"search","arguments":{"query":"x"}}</tool_call>',
        },
      ],
      finishReason: createFinishReason("stop"),
      usage,
      warnings: [],
    };
    const model = createModel({
      generateResult,
      fallback: "gemma4",
    });

    const result = await model.doGenerate({
      prompt: [],
      headers: undefined,
      abortSignal: undefined,
    });

    expect(result).toEqual(generateResult);
  });

  it("is a no-op when forced fallback format does not match content in doStream", async () => {
    const streamParts: LanguageModelV3StreamPart[] = [
      { type: "stream-start", warnings: [] },
      { type: "text-start", id: "txt-0" },
      {
        type: "text-delta",
        id: "txt-0",
        delta: '<tool_call>{"name":"search","arguments":{"query":"x"}}</tool_call>',
      },
      { type: "text-end", id: "txt-0" },
      { type: "finish", finishReason: createFinishReason("stop"), usage },
    ];
    const model = createModel({
      streamParts,
      fallback: "gemma4",
    });

    const result = await model.doStream({
      prompt: [],
      headers: undefined,
      abortSignal: undefined,
    });
    const parts = await readStreamParts(result.stream);

    expect(parts).toEqual(streamParts);
  });

  it("flushes buffered text unchanged when stream ends without finish", async () => {
    const streamParts: LanguageModelV3StreamPart[] = [
      { type: "stream-start", warnings: [] },
      { type: "text-start", id: "txt-0" },
      {
        type: "text-delta",
        id: "txt-0",
        delta:
          'content with marker-looking text <tool_call>{"name":"search","arguments":{"query":"x"}}</tool_call>',
      },
      { type: "text-end", id: "txt-0" },
    ];
    const model = createModel({ streamParts });

    const result = await model.doStream({
      prompt: [],
      headers: undefined,
      abortSignal: undefined,
    });
    const parts = await readStreamParts(result.stream);

    expect(parts).toEqual(streamParts);
  });

  it("preserves non-parsable marker-like plain text in doGenerate", async () => {
    const generateResult: LanguageModelV3GenerateResult = {
      content: [
        {
          type: "text",
          text: 'Literal docs: <tool_call>{not valid json}</tool_call> should remain unchanged.',
        },
      ],
      finishReason: createFinishReason("stop"),
      usage,
      warnings: [],
    };
    const model = createModel({ generateResult });

    const result = await model.doGenerate({
      prompt: [],
      headers: undefined,
      abortSignal: undefined,
    });

    expect(result).toEqual(generateResult);
  });

  it("preserves ordering of non-text stream events around fallback text/tool transformations", async () => {
    const model = createModel({
      streamParts: [
        { type: "stream-start", warnings: [] },
        {
          type: "response-metadata",
          id: "resp-42",
          modelId: "test-model",
          timestamp: new Date(42),
        },
        { type: "text-start", id: "txt-0" },
        {
          type: "text-delta",
          id: "txt-0",
          delta: 'Before <tool_call>{"name":"search","arguments":{"query":"x"}}</tool_call> After',
        },
        { type: "text-end", id: "txt-0" },
        { type: "finish", finishReason: createFinishReason("stop"), usage },
      ],
    });

    const result = await model.doStream({
      prompt: [],
      headers: undefined,
      abortSignal: undefined,
    });
    const parts = await readStreamParts(result.stream);

    expect(parts[0]).toEqual({ type: "stream-start", warnings: [] });
    expect(parts[1]).toEqual({
      type: "response-metadata",
      id: "resp-42",
      modelId: "test-model",
      timestamp: new Date(42),
    });
    expect(parts[parts.length - 1]).toMatchObject({
      type: "finish",
      finishReason: { unified: "tool-calls" },
    });
  });

  it("extracts StepFun-style function equals tool calls from thought blocks", async () => {
    const model = createModel({
      generateResult: {
        content: [
          {
            type: "text",
            text: `<think>Let me locate the exact line number of search_tools. I'll grep:<tool_call>
<function=grep>
<parameter=-n>
True
</parameter>
<parameter=output>
content
</parameter>
<parameter=path>
/home/buck/Projects/homelab/argocd-apps/apps/ai-tools/litellm/litellm.yaml
</parameter>
<parameter=pattern>
^          search_tools:
</parameter>
</function>
</tool_call></think>Done`,
          },
        ],
        finishReason: createFinishReason("stop"),
        usage,
        warnings: [],
      },
    });

    const result = await model.doGenerate({
      prompt: [],
      headers: undefined,
      abortSignal: undefined,
    });

    expect(result.finishReason.unified).toBe("tool-calls");
    expect(result.content).toContainEqual({
      type: "tool-call",
      toolCallId: "tc_0_grep",
      toolName: "grep",
      input:
        '{"-n":true,"output":"content","path":"/home/buck/Projects/homelab/argocd-apps/apps/ai-tools/litellm/litellm.yaml","pattern":"^          search_tools:"}',
    });
    expect(result.content).toContainEqual({ type: "text", text: "Done" });
  });
});
