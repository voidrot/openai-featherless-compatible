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
} from "./tool-call-parser";
import { ToolCallFallbackTransformStream } from "./stream";

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

    // Find text content
    const textContent = result.content.find(
      (c): c is { type: "text"; text: string } => c.type === "text",
    );

    if (!textContent || !textContent.text) {
      return result;
    }

    // Check if there are no tool calls but content has markers
    const hasToolCalls = result.content.some((c) => c.type === "tool-call");
    if (hasToolCalls) {
      return result;
    }

    // Parse tool calls from text
    const forceFormat =
      this.toolCallFallback === "auto" ? undefined : this.toolCallFallback;
    const detection = detectAndParseToolCalls(textContent.text, forceFormat);

    if (detection.toolCalls.length === 0) {
      return result;
    }

    const parsedToolCalls = coerceParsedToolCalls(
      detection.toolCalls,
      toolSchemas,
    );

    // Build new content array with parsed tool calls
    const toolCallContents: LanguageModelV3Content[] = parsedToolCalls.map(
      (tc, index) => ({
        type: "tool-call" as const,
        toolCallId: this.makeToolCallId(index, tc.toolName),
        toolName: tc.toolName,
        input: JSON.stringify(tc.arguments),
      }),
    );

    // Replace text content with cleaned version + tool calls
    const newContent: LanguageModelV3Content[] = [];

    let injectedFallbackContent = false;

    for (const part of result.content) {
      if (part.type === "text") {
        if (injectedFallbackContent) {
          newContent.push(part);
          continue;
        }

        injectedFallbackContent = true;

        // Add cleaned text (only if there's content after stripping)
        if (detection.cleanedContent.trim()) {
          newContent.push({ type: "text", text: detection.cleanedContent });
        }

        // Add tool calls
        newContent.push(...toolCallContents);
      } else {
        newContent.push(part);
      }
    }

    return {
      ...result,
      content: newContent,
      finishReason: {
        ...result.finishReason,
        unified: "tool-calls",
      },
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
