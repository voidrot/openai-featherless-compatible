import type {
  LanguageModelV3,
  LanguageModelV3CallOptions,
  LanguageModelV3GenerateResult,
  LanguageModelV3StreamResult,
  LanguageModelV3Content,
  LanguageModelV3StreamPart,
} from "@ai-sdk/provider";
import type {
  OpenAICompatibleProvider,
  OpenAICompatibleChatLanguageModel,
} from "@ai-sdk/openai-compatible";
import {
  buildToolSchemaMap,
  coerceParsedToolCalls,
  detectAndParseToolCalls,
  type FeatherlessCompatibleToolCallFallbackMode,
} from "./tool-call-parser.js";
import { ToolCallFallbackTransformStream } from "./stream.js";

/**
 * A LanguageModelV3 that wraps an OpenAI-compatible model with
 * Featherless-compatible tool call parsing fallback.
 *
 * When the API returns empty `tool_calls` but the content contains
 * tool call markers (e.g., ``), this model automatically parses
 * them and injects them into the result.
 */
export class FeatherlessCompatibleChatLanguageModel implements LanguageModelV3 {
  readonly specificationVersion = "v3" as const;
  readonly provider: string;
  readonly modelId: string;
  private readonly toolCallFallback: FeatherlessCompatibleToolCallFallbackMode;
  private readonly generateToolCallId?: (
    index: number,
    toolName: string,
  ) => string;

  private inner: OpenAICompatibleChatLanguageModel;

  constructor(
    baseProvider: OpenAICompatibleProvider,
    modelId: string,
    toolCallFallback: FeatherlessCompatibleToolCallFallbackMode,
    generateToolCallId?: (index: number, toolName: string) => string,
  ) {
    this.modelId = modelId;
    this.toolCallFallback = toolCallFallback;
    this.generateToolCallId = generateToolCallId;
    this.provider = `${baseProvider.name}.featherless`;

    // Create the inner model using the provider's languageModel factory
    this.inner = baseProvider.languageModel(
      modelId,
    ) as OpenAICompatibleChatLanguageModel;
  }

  get supportedUrls(): Record<string, RegExp[]> {
    const urls = this.inner.supportedUrls;
    return urls as Record<string, RegExp[]>;
  }

  /**
   * Generate a tool call ID, using custom function if provided.
   */
  private makeToolCallId(index: number, toolName: string): string {
    if (this.generateToolCallId) {
      return this.generateToolCallId(index, toolName);
    }
    return `tc_${index}_${toolName}`;
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

  /**
   * Override doGenerate to apply tool call fallback.
   */
  async doGenerate(
    options: LanguageModelV3CallOptions,
  ): Promise<LanguageModelV3GenerateResult> {
    const result = await this.inner.doGenerate(options);

    // If fallback is disabled, return as-is
    if (this.toolCallFallback === "disabled") {
      return result;
    }

    const toolSchemas = buildToolSchemaMap(options.tools);
    const availableToolNames = [...toolSchemas.keys()];
    const availableToolNameSet = new Set(availableToolNames);

    const textParts = result.content.filter(
      (c): c is { type: "text"; text: string } => c.type === "text",
    );

    if (textParts.length === 0) {
      return result;
    }

    // Check if there are no tool calls but content has markers
    const hasToolCalls = result.content.some((c) => c.type === "tool-call");
    if (hasToolCalls) {
      return result;
    }

    const forceFormat =
      this.toolCallFallback === "auto" ? undefined : this.toolCallFallback;
    const detections = textParts.map((part) =>
      detectAndParseToolCalls(part.text, forceFormat),
    );
    const parsedToolCallCount = detections.reduce(
      (count, detection) => count + detection.toolCalls.length,
      0,
    );
    const extractedReasoningCount = detections.reduce(
      (count, detection) =>
        count + (detection.reasoningContent?.trim() ? 1 : 0),
      0,
    );

    if (parsedToolCallCount === 0 && extractedReasoningCount === 0) {
      return result;
    }

    const newContent: LanguageModelV3Content[] = [];
    const hasReasoningContent = result.content.some(
      (part) => part.type === "reasoning",
    );
    let textPartIndex = 0;
    let parsedToolCallIndex = 0;
    let emittedToolCallCount = 0;

    for (const part of result.content) {
      if (part.type === "text") {
        const detection = detections[textPartIndex++];

        if (!hasReasoningContent && detection.reasoningContent?.trim()) {
          newContent.push({
            type: "reasoning",
            text: detection.reasoningContent,
          });
        }

        if (detection.cleanedContent.trim()) {
          newContent.push({ type: "text", text: detection.cleanedContent });
        }

        if (detection.toolCalls.length > 0) {
          const parsedToolCalls = coerceParsedToolCalls(
            detection.toolCalls,
            toolSchemas,
          );

          const invalidToolNames =
            availableToolNameSet.size === 0
              ? []
              : [
                  ...new Set(
                    parsedToolCalls
                      .filter(
                        (toolCall) =>
                          !availableToolNameSet.has(toolCall.toolName),
                      )
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
          if (nudge) {
            newContent.push({ type: "text", text: nudge });
          }

          for (const toolCall of validToolCalls) {
            newContent.push({
              type: "tool-call",
              toolCallId: this.makeToolCallId(
                parsedToolCallIndex++,
                toolCall.toolName,
              ),
              toolName: toolCall.toolName,
              input: JSON.stringify(toolCall.arguments),
            });
            emittedToolCallCount++;
          }
        }
      } else {
        newContent.push(part);
      }
    }

    return {
      ...result,
      content: newContent,
      ...(emittedToolCallCount > 0
        ? {
            finishReason: {
              ...result.finishReason,
              unified: "tool-calls",
            },
          }
        : {}),
    };
  }

  /**
   * Override doStream to apply tool call fallback on streaming responses.
   */
  async doStream(
    options: LanguageModelV3CallOptions,
  ): Promise<LanguageModelV3StreamResult> {
    const innerResult = await this.inner.doStream(options);

    // If fallback is disabled, return as-is
    if (this.toolCallFallback === "disabled") {
      return innerResult;
    }

    const toolSchemas = buildToolSchemaMap(options.tools);

    // Wrap the text stream to detect and parse tool calls
    const textStream = innerResult.stream.pipeThrough(
      new ToolCallFallbackTransformStream(
        this.toolCallFallback,
        (index, toolName) => this.makeToolCallId(index, toolName),
        toolSchemas,
      ),
    );

    return {
      ...innerResult,
      stream: textStream,
    };
  }
}
