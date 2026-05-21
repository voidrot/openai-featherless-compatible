export {
  createFeatherlessCompatibleProvider,
  type FeatherlessCompatibleProviderSettings,
  type FeatherlessCompatibleProvider,
} from "./provider";

export { FeatherlessCompatibleChatLanguageModel } from "./model";

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
} from "./tool-call-parser";

export { ToolCallFallbackTransformStream } from "./stream";
