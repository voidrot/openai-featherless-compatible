import type { LanguageModelV3StreamPart } from "@ai-sdk/provider";
import {
  coerceParsedToolCalls,
  detectAndParseToolCalls,
  type FeatherlessCompatibleToolCallFallbackMode,
  type ToolSchemaMap,
} from "./tool-call-parser";

/**
 * A TransformStream that wraps the text stream from an OpenAI-compatible model
 * and detects Featherless-compatible tool call markers in the text content.
 *
 * When tool calls are found in the text, they are emitted as tool-call stream
 * parts at the end of the stream, and the text content is cleaned.
 *
 * This handles the streaming case where the model outputs tool calls as text
 * within markers instead of using native function calling.
 */
export class ToolCallFallbackTransformStream extends TransformStream<
  LanguageModelV3StreamPart,
  LanguageModelV3StreamPart
> {
  private buffer = "";
  private readonly bufferedTextParts: LanguageModelV3StreamPart[] = [];
  private textPartId: string | undefined;
  private sawNativeToolCalls = false;
  private readonly seenReasoningIds = new Set<string>();
  private fallbackReasoningIndex = 0;
  private readonly toolCallFallback: FeatherlessCompatibleToolCallFallbackMode;
  private readonly makeToolCallId: (index: number, toolName: string) => string;
  private readonly toolSchemas: ToolSchemaMap;

  constructor(
    toolCallFallback: FeatherlessCompatibleToolCallFallbackMode,
    makeToolCallId: (index: number, toolName: string) => string,
    toolSchemas: ToolSchemaMap,
  ) {
    super({
      transform: (chunk, controller) => {
        this.transformChunk(chunk, controller);
      },
      flush: (controller) => {
        this.flushChunk(controller);
      },
    });

    this.toolCallFallback = toolCallFallback;
    this.makeToolCallId = makeToolCallId;
    this.toolSchemas = toolSchemas;
  }

  private transformChunk(
    chunk: LanguageModelV3StreamPart,
    controller: TransformStreamDefaultController<LanguageModelV3StreamPart>,
  ): void {
    if (this.toolCallFallback === "disabled") {
      controller.enqueue(chunk);
      return;
    }

    if (this.sawNativeToolCalls) {
      controller.enqueue(chunk);
      return;
    }

    switch (chunk.type) {
      case "text-start":
        this.textPartId ??= chunk.id;
        this.bufferedTextParts.push(chunk);
        return;

      case "text-delta":
        this.textPartId ??= chunk.id;
        this.buffer += chunk.delta;
        this.bufferedTextParts.push(chunk);
        return;

      case "text-end":
        this.textPartId ??= chunk.id;
        this.bufferedTextParts.push(chunk);
        return;

      case "tool-input-start":
      case "tool-input-delta":
      case "tool-input-end":
      case "tool-call":
        this.sawNativeToolCalls = true;
        this.emitBufferedText(controller);
        controller.enqueue(chunk);
        return;

      case "reasoning-start":
        this.seenReasoningIds.add(chunk.id);
        controller.enqueue(chunk);
        return;

      case "finish": {
        const emittedFallback = this.emitFallbackToolCalls(controller);
        controller.enqueue(
          emittedFallback
            ? {
                ...chunk,
                finishReason: {
                  ...chunk.finishReason,
                  unified: "tool-calls",
                },
              }
            : chunk,
        );
        return;
      }

      default:
        controller.enqueue(chunk);
    }
  }

  private emitBufferedText(
    controller: TransformStreamDefaultController<LanguageModelV3StreamPart>,
  ): void {
    for (const part of this.bufferedTextParts) {
      controller.enqueue(part);
    }

    this.resetBuffer();
  }

  private emitFallbackToolCalls(
    controller: TransformStreamDefaultController<LanguageModelV3StreamPart>,
  ): boolean {
    if (this.sawNativeToolCalls || this.buffer.length === 0) {
      this.emitBufferedText(controller);
      return false;
    }

    const forceFormat =
      this.toolCallFallback === "auto" || this.toolCallFallback === "disabled"
        ? undefined
        : this.toolCallFallback;
    const detection = detectAndParseToolCalls(this.buffer, forceFormat);

    if (detection.toolCalls.length === 0) {
      this.emitBufferedText(controller);
      return false;
    }

    const parsedToolCalls = coerceParsedToolCalls(
      detection.toolCalls,
      this.toolSchemas,
    );
    const availableToolNames = [...this.toolSchemas.keys()];
    const availableToolNameSet = new Set(availableToolNames);
    const invalidToolNames =
      availableToolNameSet.size === 0
        ? []
        : [
            ...new Set(
              parsedToolCalls
                .filter((toolCall) => !availableToolNameSet.has(toolCall.toolName))
                .map((toolCall) => toolCall.toolName),
            ),
          ];
    const validToolCalls =
      availableToolNameSet.size === 0
        ? parsedToolCalls
        : parsedToolCalls.filter((toolCall) =>
            availableToolNameSet.has(toolCall.toolName),
          );
    const nudge = this.createInvalidToolNameNudge(
      invalidToolNames,
      availableToolNames,
    );

    if (detection.reasoningContent?.trim()) {
      const reasoningId = this.makeFallbackReasoningId();
      controller.enqueue({ type: "reasoning-start", id: reasoningId });
      controller.enqueue({
        type: "reasoning-delta",
        id: reasoningId,
        delta: detection.reasoningContent,
      });
      controller.enqueue({ type: "reasoning-end", id: reasoningId });
    }

    const textId = this.textPartId ?? "txt-fallback-0";
    const hasCleanedText = detection.cleanedContent.trim().length > 0;
    const combinedText = hasCleanedText
      ? `${detection.cleanedContent}${nudge ? `\n${nudge}` : ""}`
      : nudge;
    if (combinedText) {
      controller.enqueue({ type: "text-start", id: textId });
      controller.enqueue({
        type: "text-delta",
        id: textId,
        delta: combinedText,
      });
      controller.enqueue({ type: "text-end", id: textId });
    }

    for (const [index, toolCall] of validToolCalls.entries()) {
      const toolCallId = this.makeToolCallId(index, toolCall.toolName);
      const input = JSON.stringify(toolCall.arguments);

      controller.enqueue({
        type: "tool-input-start",
        id: toolCallId,
        toolName: toolCall.toolName,
      });
      controller.enqueue({
        type: "tool-input-delta",
        id: toolCallId,
        delta: input,
      });
      controller.enqueue({
        type: "tool-input-end",
        id: toolCallId,
      });
      controller.enqueue({
        type: "tool-call",
        toolCallId,
        toolName: toolCall.toolName,
        input,
      });
    }

    this.resetBuffer();
    return validToolCalls.length > 0;
  }

  private flushChunk(
    controller: TransformStreamDefaultController<LanguageModelV3StreamPart>,
  ): void {
    if (!this.sawNativeToolCalls && this.bufferedTextParts.length > 0) {
      this.emitBufferedText(controller);
    }
  }

  private resetBuffer(): void {
    this.buffer = "";
    this.bufferedTextParts.length = 0;
    this.textPartId = undefined;
  }

  private createInvalidToolNameNudge(
    invalidToolNames: string[],
    availableToolNames: string[],
  ): string {
    if (invalidToolNames.length === 0 || availableToolNames.length === 0) {
      return "";
    }

    const invalid = invalidToolNames.join(", ");
    const available = availableToolNames.join(", ");
    const noun = invalidToolNames.length === 1 ? "tool name" : "tool names";

    return `Invalid ${noun}: ${invalid}. Available tools: ${available}. Retry using one of the available tool names.`;
  }

  private makeFallbackReasoningId(): string {
    while (true) {
      const id = `reasoning-fallback-${this.fallbackReasoningIndex++}`;
      if (!this.seenReasoningIds.has(id)) {
        this.seenReasoningIds.add(id);
        return id;
      }
    }
  }
}
