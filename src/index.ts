export {
  createFeatherlessCompatibleProvider,
  type FeatherlessCompatibleProviderSettings,
  type FeatherlessCompatibleProvider,
} from "./provider.js";

export { FeatherlessCompatibleChatLanguageModel } from "./model.js";

export {
  detectAndParseToolCalls,
  stripToolCallMarkers,
  buildToolSchemaMap,
  coerceParsedToolCalls,
  generateToolCallId,
  type ParsedToolCall,
  type ToolCallDetection,
  type FeatherlessCompatibleToolCallFormat,
  type FeatherlessCompatibleToolCallFallbackMode,
} from "./tool-call-parser.js";

export { ToolCallFallbackTransformStream } from "./stream.js";
