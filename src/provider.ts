import type {
  ProviderV3,
  LanguageModelV3,
  EmbeddingModelV3,
  ImageModelV3,
} from "@ai-sdk/provider";
import { createOpenAICompatible } from "@ai-sdk/openai-compatible";
import type {
  OpenAICompatibleProviderSettings,
  OpenAICompatibleProvider,
} from "@ai-sdk/openai-compatible";
import { FeatherlessCompatibleChatLanguageModel } from "./model.js";
import type { FeatherlessCompatibleToolCallFallbackMode } from "./tool-call-parser.js";

export interface FeatherlessCompatibleProviderSettings extends OpenAICompatibleProviderSettings {
  /**
   * Tool call fallback mode.
   * - `'auto'`: Auto-detect tool call format from response text (default)
   * - `'hermes'`, `'gemma4'`, `'kimi-k2'`, `'deepseek-v3'`, `'mistral'`,
   *   `'minimax'`, `'dsml'`, `'xml'`: Force a specific fallback parser
   * - `'disabled'`: Skip tool call fallback entirely
   */
  toolCallFallback?: FeatherlessCompatibleToolCallFallbackMode;

  /**
   * Custom function to generate tool call IDs.
   * Default: `tc_${index}_${toolName}`
   */
  generateToolCallId?: (index: number, toolName: string) => string;
}

export interface FeatherlessCompatibleProvider extends ProviderV3 {
  (modelId: string): LanguageModelV3;

  readonly specificationVersion: "v3";
  readonly toolCallFallback: FeatherlessCompatibleToolCallFallbackMode;

  languageModel(modelId: string): LanguageModelV3;
  chatModel(modelId: string): LanguageModelV3;
  completionModel(modelId: string): LanguageModelV3;
  embeddingModel(modelId: string): EmbeddingModelV3;
  textEmbeddingModel(modelId: string): EmbeddingModelV3;
  imageModel(modelId: string): ImageModelV3;
}

/**
 * Create a Featherless-compatible provider that wraps `@ai-sdk/openai-compatible`
 * with client-side tool call parsing fallback.
 *
 * When a model outputs tool calls as text within XML markers instead of
 * using native function calling, the provider automatically detects and
 * parses them from the response content.
 *
 * @example
 * ```ts
 * const featherless = createFeatherlessCompatibleProvider({
 *   name: 'my-featherless',
 *   baseURL: 'https://api.example.com/v1',
 *   apiKey: process.env.API_KEY,
 *   toolCallFallback: 'auto',
 * });
 *
 * const model = featherless('featherless-3-7b');
 * ```
 */
export function createFeatherlessCompatibleProvider(
  options: FeatherlessCompatibleProviderSettings,
): FeatherlessCompatibleProvider {
  const fallback = options.toolCallFallback ?? "auto";
  const generateToolCallId = options.generateToolCallId;

  // Create the underlying openai-compatible provider
  const innerProvider = createOpenAICompatible({
    ...options,
    name: options.name,
  });

  function createLanguageModel(modelId: string): LanguageModelV3 {
    return new FeatherlessCompatibleChatLanguageModel(
      innerProvider as OpenAICompatibleProvider,
      modelId,
      fallback,
      generateToolCallId,
    );
  }

  const provider: FeatherlessCompatibleProvider = Object.assign(createLanguageModel, {
    specificationVersion: "v3" as const,
    toolCallFallback: fallback,
    languageModel: createLanguageModel,
    chatModel: createLanguageModel,
    completionModel: createLanguageModel,
    embeddingModel: (modelId: string) => innerProvider.embeddingModel(modelId),
    textEmbeddingModel: (modelId: string) =>
      innerProvider.textEmbeddingModel?.(modelId) ??
      innerProvider.embeddingModel(modelId),
    imageModel: (modelId: string) => innerProvider.imageModel(modelId),
  }) as FeatherlessCompatibleProvider;

  return provider;
}
