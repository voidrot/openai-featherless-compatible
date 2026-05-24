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

    expect(parts).toContainEqual({
      type: "reasoning-start",
      id: "reasoning-0",
    });
    expect(parts).toContainEqual({
      type: "reasoning-delta",
      id: "reasoning-0",
      delta: "step-by-step",
    });
    expect(parts).toContainEqual({ type: "reasoning-end", id: "reasoning-0" });
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
});
