import type {
  JSONSchema7,
  LanguageModelV3FunctionTool,
  LanguageModelV3ProviderTool,
} from "@ai-sdk/provider";

/**
 * Featherless-compatible tool call parser.
 *
 * Extracts tool calls from raw text content when models output tool calls
 * as text within XML-style markers instead of using native function calling.
 *
 * Supports multiple formats:
 * - Fenced JSON: ```json { ... } ```
 * - Gemma 4: <|tool_call> ... <|end_of_tool_call|>
 * - Kimi K2: <|tool_calls_section_begin|> ... <|tool_calls_section_end|>
 * - DeepSeek V3: <｜tool▁calls▁begin｜> ... <｜tool▁calls▁end｜>
 * - Mistral: [TOOL_CALLS]
 * - MiniMax: <minimax:tool_call> ... </minimax:tool_call>
 * - DeepSeek V4 (DSML): <dsml:tool_call> ... </dsml:tool_call>
 * - Generic XML: <tool_call>, <tool_calls>, <function_call>, <function_calls>,
 *   and Gemma/OpenClaw-style <function name="..."> ... </function>
 *
 * The cleaned content also strips provider-style reasoning tags so the
 * fallback path does not leak chain-of-thought text into the visible response.
 */

export type FeatherlessCompatibleToolCallFormat =
  | "fenced-json"
  | "gemma4"
  | "kimi-k2"
  | "deepseek-v3"
  | "mistral"
  | "minimax"
  | "dsml"
  | "xml";

export type FeatherlessCompatibleToolCallFallbackMode =
  | "auto"
  | FeatherlessCompatibleToolCallFormat
  | "hermes"
  | "disabled";

export interface ParsedToolCall {
  toolCallId: string;
  toolName: string;
  arguments: Record<string, unknown>;
}

export interface ToolCallDetection {
  format: FeatherlessCompatibleToolCallFormat | null;
  toolCalls: ParsedToolCall[];
  cleanedContent: string;
  reasoningContent?: string;
}

export type ToolSchemaMap = Map<string, JSONSchema7>;

interface FormatHandler {
  name: FeatherlessCompatibleToolCallFormat;
  detect: (content: string) => boolean;
  parse: (content: string) => ParsedToolCall[];
  strip: (content: string) => string;
}

const MISTRAL_MARKER = "[TOOL_CALLS]";
const MINIMAX_OPEN = "<minimax:tool_call>";
const MINIMAX_CLOSE = "</minimax:tool_call>";
const DSML_OPEN = "<dsml:tool_call>";
const DSML_CLOSE = "</dsml:tool_call>";
const DSML_ALT_OPEN = "<｜｜DSML｜｜tool_calls>";
const DSML_ALT_CLOSE = "</｜｜DSML｜｜tool_calls>";
const KIMI_OPEN = "<|tool_calls_section_begin|>";
const KIMI_CLOSE = "<|tool_calls_section_end|>";
const DEEPSEEK_V3_OPEN = "<｜tool▁calls▁begin｜>";
const DEEPSEEK_V3_CLOSE = "<｜tool▁calls▁end｜>";
const GEMMA4_OPEN = "<|tool_call>";
const GEMMA4_CLOSE = "<|end_of_tool_call|>";
const FENCED_JSON_BLOCK_RE = /```(?:json)?\s*([\s\S]*?)```/gi;
const XML_TOOL_BLOCK_RE =
  /<(tool_call|tool_calls|function_call|function_calls)\b[^>]*>([\s\S]*?)<\/\1>/gi;
const XML_FUNCTION_NAME_RE =
  /<function\b[^>]*\bname\s*=\s*"([^"]+)"[^>]*>([\s\S]*?)<\/function>/gi;
const XML_FUNCTION_EQUALS_RE =
  /<function\s*=\s*([^>\s]+)[^>]*>([\s\S]*?)<\/function>/gi;

export function generateToolCallId(index: number, toolName: string): string {
  return `tc_${index}_${toolName}`;
}

export function buildToolSchemaMap(
  tools?: Array<LanguageModelV3FunctionTool | LanguageModelV3ProviderTool>,
): ToolSchemaMap {
  const schemas: ToolSchemaMap = new Map();

  for (const tool of tools ?? []) {
    if (tool.type === "function") {
      schemas.set(tool.name, tool.inputSchema);
    }
  }

  return schemas;
}

export function coerceParsedToolCalls(
  toolCalls: ParsedToolCall[],
  toolSchemas?: ToolSchemaMap,
): ParsedToolCall[] {
  if (!toolSchemas || toolSchemas.size === 0) {
    return toolCalls;
  }

  return toolCalls.map((toolCall) => ({
    ...toolCall,
    arguments: coerceArgumentsToSchema(
      toolCall.arguments,
      toolSchemas.get(toolCall.toolName),
    ),
  }));
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

function safeJsonParse(text: string): unknown | undefined {
  try {
    return JSON.parse(text);
  } catch {
    return undefined;
  }
}

function normalizeLooseJson(text: string): string[] {
  const trimmed = text.trim();
  const candidates = new Set<string>();

  if (!trimmed) {
    return [];
  }

  candidates.add(trimmed);

  if (trimmed.startsWith("```") && trimmed.endsWith("```")) {
    const withoutFences = trimmed
      .replace(/^```(?:json)?\s*/i, "")
      .replace(/```$/, "")
      .trim();
    if (withoutFences) {
      candidates.add(withoutFences);
    }
  }

  const pythonLiterals = trimmed
    .replace(/\bNone\b/g, "null")
    .replace(/\bTrue\b/g, "true")
    .replace(/\bFalse\b/g, "false");
  candidates.add(pythonLiterals);

  const withoutTrailingCommas = pythonLiterals.replace(/,\s*([}\]])/g, "$1");
  candidates.add(withoutTrailingCommas);

  const singleQuoted = withoutTrailingCommas
    .replace(/([{,]\s*)'([^'\\]*(?:\\.[^'\\]*)*)'\s*:/g, '$1"$2":')
    .replace(/:\s*'([^'\\]*(?:\\.[^'\\]*)*)'/g, ': "$1"');
  candidates.add(singleQuoted);

  return [...candidates];
}

function tryParseLooseJson(text: string): unknown | undefined {
  for (const candidate of normalizeLooseJson(text)) {
    const parsed = safeJsonParse(candidate);
    if (parsed !== undefined) {
      return parsed;
    }
  }

  const balanced = findFirstBalancedJsonSegment(text);
  if (balanced && balanced.trim() !== text.trim()) {
    for (const candidate of normalizeLooseJson(balanced)) {
      const parsed = safeJsonParse(candidate);
      if (parsed !== undefined) {
        return parsed;
      }
    }
  }

  return undefined;
}

function findFirstBalancedJsonSegment(text: string): string | undefined {
  let start = -1;
  let depth = 0;
  let inString = false;
  let escaped = false;

  for (let index = 0; index < text.length; index++) {
    const char = text[index];

    if (start === -1) {
      if (char === "{" || char === "[") {
        start = index;
        depth = 1;
      }
      continue;
    }

    if (escaped) {
      escaped = false;
      continue;
    }

    if (char === "\\") {
      escaped = true;
      continue;
    }

    if (char === '"') {
      inString = !inString;
      continue;
    }

    if (inString) {
      continue;
    }

    if (char === "{" || char === "[") {
      depth++;
      continue;
    }

    if (char === "}" || char === "]") {
      depth--;
      if (depth === 0) {
        return text.slice(start, index + 1);
      }
    }
  }

  return undefined;
}

function extractBetweenTags(
  content: string,
  openTag: string,
  closeTag: string,
): string[] {
  const results: string[] = [];
  let offset = 0;

  while (true) {
    const openIndex = content.indexOf(openTag, offset);
    if (openIndex === -1) {
      break;
    }

    const closeIndex = content.indexOf(closeTag, openIndex + openTag.length);
    if (closeIndex === -1) {
      break;
    }

    results.push(content.slice(openIndex + openTag.length, closeIndex));
    offset = closeIndex + closeTag.length;
  }

  return results;
}

function stripBetweenTags(
  content: string,
  openTag: string,
  closeTag: string,
): string {
  let result = content;
  let offset = 0;

  while (true) {
    const openIndex = result.indexOf(openTag, offset);
    if (openIndex === -1) {
      return result;
    }

    const closeIndex = result.indexOf(closeTag, openIndex + openTag.length);
    if (closeIndex === -1) {
      return result;
    }

    result =
      result.slice(0, openIndex) + result.slice(closeIndex + closeTag.length);
    offset = openIndex;
  }
}

function stripWithRegex(content: string, pattern: RegExp): string {
  return content.replace(pattern, "");
}

function parseJsonToolCalls(text: string): ParsedToolCall[] {
  const parsed = tryParseLooseJson(text);
  if (parsed === undefined) {
    return [];
  }

  return extractToolCallsFromPayload(parsed);
}

function extractToolCallsFromPayload(payload: unknown): ParsedToolCall[] {
  if (Array.isArray(payload)) {
    return payload.flatMap((item) => extractToolCallsFromPayload(item));
  }

  if (!isRecord(payload)) {
    return [];
  }

  const grouped = payload.tool_calls ?? payload.function_calls ?? payload.calls;
  if (Array.isArray(grouped)) {
    return grouped.flatMap((item) => extractToolCallsFromPayload(item));
  }

  const toolName = extractToolName(payload);
  if (!toolName) {
    return [];
  }

  const argumentsObject = extractToolArguments(payload);
  if (argumentsObject === undefined) {
    return [];
  }

  return [
    {
      toolCallId: generateToolCallId(0, toolName),
      toolName,
      arguments: argumentsObject,
    },
  ];
}

function extractToolName(payload: Record<string, unknown>): string | undefined {
  if (typeof payload.name === "string" && payload.name.length > 0) {
    return payload.name;
  }

  if (typeof payload.tool_name === "string" && payload.tool_name.length > 0) {
    return payload.tool_name;
  }

  if (
    isRecord(payload.function) &&
    typeof payload.function.name === "string" &&
    payload.function.name.length > 0
  ) {
    return payload.function.name;
  }

  return undefined;
}

function extractToolArguments(
  payload: Record<string, unknown>,
): Record<string, unknown> | undefined {
  const directArguments =
    payload.arguments ?? payload.input ?? payload.parameters ?? payload.args;
  const directParsed = parseToolArguments(directArguments);
  if (directParsed !== undefined) {
    return directParsed;
  }

  if (isRecord(payload.function)) {
    const nestedArguments =
      payload.function.arguments ?? payload.function.input;
    const nestedParsed = parseToolArguments(nestedArguments);
    if (nestedParsed !== undefined) {
      return nestedParsed;
    }
  }

  return {};
}

function parseToolArguments(
  value: unknown,
): Record<string, unknown> | undefined {
  if (value === undefined) {
    return undefined;
  }

  if (isRecord(value)) {
    return value;
  }

  if (typeof value === "string") {
    const parsed = tryParseLooseJson(value);
    if (isRecord(parsed)) {
      return parsed;
    }
  }

  return undefined;
}

function parseParameterTags(
  content: string,
): Record<string, unknown> | undefined {
  const params: Record<string, unknown> = {};
  let found = false;

  for (const match of content.matchAll(
    /<parameter\b[^>]*\bname="([^"]+)"[^>]*>([\s\S]*?)<\/parameter>/g,
  )) {
    found = true;
    params[match[1]] = parseParameterValue(match[2].trim());
  }

  for (const match of content.matchAll(
    /<parameter\s*=\s*([^>\s]+)[^>]*>([\s\S]*?)<\/parameter>/g,
  )) {
    found = true;
    params[match[1]] = parseParameterValue(match[2].trim());
  }

  return found ? params : undefined;
}

function parseParameterValue(value: string): unknown {
  const parsed = tryParseLooseJson(value);
  if (parsed !== undefined) {
    return parsed;
  }

  if (/^-?\d+$/.test(value)) {
    return Number(value);
  }

  if (/^-?\d+\.\d+$/.test(value)) {
    return Number(value);
  }

  if (value === "true") {
    return true;
  }

  if (value === "false") {
    return false;
  }

  return value;
}

function reindexToolCalls(toolCalls: ParsedToolCall[]): ParsedToolCall[] {
  return toolCalls.map((toolCall, index) => ({
    ...toolCall,
    toolCallId: generateToolCallId(index, toolCall.toolName),
  }));
}

function normalizePotentialJsonString(value: unknown): unknown {
  if (typeof value !== "string") {
    return value;
  }

  const parsed = tryParseLooseJson(value);
  return parsed === undefined ? value : parsed;
}

function coerceArgumentsToSchema(
  argumentsObject: Record<string, unknown>,
  schema?: JSONSchema7,
): Record<string, unknown> {
  const coerced = coerceValueToSchema(argumentsObject, schema);
  return isRecord(coerced) ? coerced : argumentsObject;
}

function coerceValueToSchema(value: unknown, schema?: JSONSchema7): unknown {
  const normalized = normalizePotentialJsonString(value);

  if (!schema) {
    return normalized;
  }

  const unionCoerced = tryUnionSchema(normalized, schema);
  if (unionCoerced !== undefined) {
    return unionCoerced;
  }

  if (schema.type === "object") {
    return coerceObjectLike(normalized, schema);
  }

  if (schema.type === "array") {
    return coerceArrayLike(normalized, schema);
  }

  if (schema.type === "integer") {
    if (typeof normalized === "number" && Number.isInteger(normalized)) {
      return normalized;
    }

    if (typeof normalized === "string" && /^-?\d+$/.test(normalized.trim())) {
      return Number.parseInt(normalized.trim(), 10);
    }

    return normalized;
  }

  if (schema.type === "number") {
    if (typeof normalized === "number") {
      return normalized;
    }

    if (
      typeof normalized === "string" &&
      /^-?\d+(?:\.\d+)?$/.test(normalized.trim())
    ) {
      return Number(normalized.trim());
    }

    return normalized;
  }

  if (schema.type === "boolean") {
    if (typeof normalized === "boolean") {
      return normalized;
    }

    if (typeof normalized === "string") {
      const lower = normalized.trim().toLowerCase();
      if (lower === "true" || lower === "1") {
        return true;
      }
      if (lower === "false" || lower === "0") {
        return false;
      }
    }

    return normalized;
  }

  if (schema.type === "string") {
    if (typeof normalized === "string") {
      return normalized;
    }

    if (normalized === null || normalized === undefined) {
      return "";
    }

    return typeof normalized === "object"
      ? JSON.stringify(normalized)
      : String(normalized);
  }

  return normalized;
}

function tryUnionSchema(
  value: unknown,
  schema: JSONSchema7,
): unknown | undefined {
  const variants = [
    ...(Array.isArray(schema.oneOf) ? schema.oneOf : []),
    ...(Array.isArray(schema.anyOf) ? schema.anyOf : []),
  ].filter(isRecord) as JSONSchema7[];

  if (variants.length === 0) {
    return undefined;
  }

  for (const variant of variants) {
    const coerced = coerceValueToSchema(value, variant);
    if (isValueCompatibleWithSchema(coerced, variant)) {
      return coerced;
    }
  }

  return undefined;
}

function isValueCompatibleWithSchema(
  value: unknown,
  schema: JSONSchema7,
): boolean {
  if (schema.type === "object") {
    return isRecord(value);
  }

  if (schema.type === "array") {
    return Array.isArray(value);
  }

  if (schema.type === "integer") {
    return typeof value === "number" && Number.isInteger(value);
  }

  if (schema.type === "number") {
    return typeof value === "number";
  }

  if (schema.type === "boolean") {
    return typeof value === "boolean";
  }

  if (schema.type === "string") {
    return typeof value === "string";
  }

  return true;
}

function coerceObjectLike(
  value: unknown,
  schema: JSONSchema7,
): Record<string, unknown> {
  const objectValue = isRecord(value) ? value : {};
  const result: Record<string, unknown> = { ...objectValue };
  const properties = isRecord(schema.properties)
    ? schema.properties
    : undefined;
  const additionalProperties = isRecord(schema.additionalProperties)
    ? (schema.additionalProperties as JSONSchema7)
    : undefined;

  if (properties) {
    for (const [key, propertySchema] of Object.entries(properties)) {
      if (key in result && isRecord(propertySchema)) {
        result[key] = coerceValueToSchema(
          result[key],
          propertySchema as JSONSchema7,
        );
      }
    }
  }

  if (additionalProperties) {
    for (const [key, propertyValue] of Object.entries(result)) {
      if (!properties || !(key in properties)) {
        result[key] = coerceValueToSchema(propertyValue, additionalProperties);
      }
    }
  }

  return result;
}

function coerceArrayLike(value: unknown, schema: JSONSchema7): unknown[] {
  const itemsSchema = isRecord(schema.items)
    ? (schema.items as JSONSchema7)
    : undefined;
  const arrayValue = Array.isArray(value)
    ? value
    : value === undefined || value === null || value === ""
      ? []
      : [value];

  return itemsSchema
    ? arrayValue.map((item) => coerceValueToSchema(item, itemsSchema))
    : arrayValue;
}

function stripReasoningBlocks(content: string): string {
  if (!content) {
    return "";
  }

  let cleaned = content;

  cleaned = cleaned.replace(/<think>[\s\S]*?<\/think>/gi, "");
  cleaned = cleaned.replace(/<thinking>[\s\S]*?<\/thinking>/gi, "");
  cleaned = cleaned.replace(/<reasoning>[\s\S]*?<\/reasoning>/gi, "");
  cleaned = cleaned.replace(
    /<REASONING_SCRATCHPAD>[\s\S]*?<\/REASONING_SCRATCHPAD>/gi,
    "",
  );
  cleaned = cleaned.replace(/<thought>[\s\S]*?<\/thought>/gi, "");

  cleaned = cleaned.replace(
    /(?:^|\n)[ \t]*<(?:think|thinking|reasoning|thought|REASONING_SCRATCHPAD)\b[^>]*>[\s\S]*$/gi,
    "",
  );

  cleaned = cleaned.replace(
    /<\/?(?:think|thinking|reasoning|thought|REASONING_SCRATCHPAD)>\s*/gi,
    "",
  );

  return cleaned;
}

function extractReasoningBlocks(content: string): string {
  if (!content) {
    return "";
  }

  const fragments: string[] = [];
  const closedBlockPattern =
    /<(think|thinking|reasoning|thought|REASONING_SCRATCHPAD)\b[^>]*>([\s\S]*?)<\/\1>/gi;

  for (const match of content.matchAll(closedBlockPattern)) {
    let block = match[2]?.trim();
    if (block) {
      // Strip tool-call markers from extracted reasoning content
      block = stripToolCallMarkers(block, "xml").trim();
      if (block) {
        fragments.push(block);
      }
    }
  }

  const openTagPattern =
    /<(think|thinking|reasoning|thought|REASONING_SCRATCHPAD)\b[^>]*>/gi;
  let lastOpenTag: RegExpExecArray | null = null;

  for (const match of content.matchAll(openTagPattern)) {
    lastOpenTag = match;
  }

  if (lastOpenTag) {
    const tagName = lastOpenTag[1];
    const rest = content.slice(lastOpenTag.index + lastOpenTag[0].length);
    const closeTagPattern = new RegExp(`</${tagName}>`, "i");

    if (!closeTagPattern.test(rest)) {
      let trailingBlock = rest.trim();
      if (trailingBlock) {
        // Strip tool-call markers from trailing reasoning content
        trailingBlock = stripToolCallMarkers(trailingBlock, "xml").trim();
        if (trailingBlock) {
          fragments.push(trailingBlock);
        }
      }
    }
  }

  return fragments.join("\n\n");
}

function detectFencedJson(content: string): boolean {
  FENCED_JSON_BLOCK_RE.lastIndex = 0;
  return FENCED_JSON_BLOCK_RE.test(content);
}

function parseFencedJson(content: string): ParsedToolCall[] {
  const toolCalls: ParsedToolCall[] = [];
  FENCED_JSON_BLOCK_RE.lastIndex = 0;

  for (const match of content.matchAll(FENCED_JSON_BLOCK_RE)) {
    toolCalls.push(...parseJsonToolCalls(match[1].trim()));
  }

  return toolCalls;
}

function stripFencedJson(content: string): string {
  FENCED_JSON_BLOCK_RE.lastIndex = 0;
  return content.replace(FENCED_JSON_BLOCK_RE, (fullMatch, block) => {
    return parseJsonToolCalls(String(block).trim()).length > 0 ? "" : fullMatch;
  });
}

function detectGemma4(content: string): boolean {
  return content.includes(GEMMA4_OPEN) || content.includes(GEMMA4_CLOSE);
}

function parseGemma4(content: string): ParsedToolCall[] {
  const blocks = extractBetweenTags(content, GEMMA4_OPEN, GEMMA4_CLOSE);
  const toolCalls: ParsedToolCall[] = [];

  for (const block of blocks) {
    const trimmed = block.trim();
    const jsonCalls = parseJsonToolCalls(trimmed);
    if (jsonCalls.length > 0) {
      toolCalls.push(...jsonCalls);
      continue;
    }

    const callMatch = trimmed.match(/^call:([\w.-]+)(?:\((.*)\)|\{(.*)\})$/s);
    if (!callMatch) {
      continue;
    }

    const toolName = callMatch[1];
    const argumentsSource = callMatch[2] ?? callMatch[3] ?? "";
    const argumentsObject = parseGemma4Arguments(argumentsSource);
    if (!argumentsObject) {
      continue;
    }

    toolCalls.push({
      toolCallId: generateToolCallId(toolCalls.length, toolName),
      toolName,
      arguments: argumentsObject,
    });
  }

  return toolCalls;
}

function parseGemma4Arguments(
  text: string,
): Record<string, unknown> | undefined {
  if (!text.trim()) {
    return {};
  }

  const parsed = tryParseLooseJson(text);
  if (isRecord(parsed)) {
    return parsed;
  }

  const pairs = splitGemma4Arguments(text);
  if (!pairs) {
    return undefined;
  }

  return Object.fromEntries(
    pairs.map(([key, value]) => [key, parseParameterValue(value)]),
  );
}

function splitGemma4Arguments(
  text: string,
): Array<[string, string]> | undefined {
  const pairs: Array<[string, string]> = [];
  let current = "";
  let depth = 0;
  let inString = false;
  let quote = "";

  for (let index = 0; index < text.length; index++) {
    const char = text[index];

    if (inString) {
      current += char;
      if (char === quote && text[index - 1] !== "\\") {
        inString = false;
      }
      continue;
    }

    if (char === '"' || char === "'") {
      inString = true;
      quote = char;
      current += char;
      continue;
    }

    if (char === "{" || char === "[") {
      depth++;
      current += char;
      continue;
    }

    if (char === "}" || char === "]") {
      depth--;
      current += char;
      continue;
    }

    if (char === "," && depth === 0) {
      const pair = parseGemma4Pair(current.trim());
      if (!pair) {
        return undefined;
      }
      pairs.push(pair);
      current = "";
      continue;
    }

    current += char;
  }

  if (current.trim()) {
    const pair = parseGemma4Pair(current.trim());
    if (!pair) {
      return undefined;
    }
    pairs.push(pair);
  }

  return pairs;
}

function parseGemma4Pair(text: string): [string, string] | undefined {
  const separatorIndex = text.includes(":")
    ? text.indexOf(":")
    : text.indexOf("=");

  if (separatorIndex === -1) {
    return undefined;
  }

  const key = text
    .slice(0, separatorIndex)
    .trim()
    .replace(/^['"]|['"]$/g, "");
  const value = text.slice(separatorIndex + 1).trim();

  return key ? [key, value] : undefined;
}

function stripGemma4(content: string): string {
  return stripBetweenTags(content, GEMMA4_OPEN, GEMMA4_CLOSE);
}

function detectKimiK2(content: string): boolean {
  return content.includes(KIMI_OPEN) || content.includes(KIMI_CLOSE);
}

function parseKimiK2(content: string): ParsedToolCall[] {
  const blocks = extractBetweenTags(content, KIMI_OPEN, KIMI_CLOSE);
  const toolCalls: ParsedToolCall[] = [];

  for (const block of blocks) {
    const lines = block
      .split("\n")
      .map((line) => line.trim())
      .filter(Boolean);
    for (const line of lines) {
      toolCalls.push(...parseJsonToolCalls(line));
    }
  }

  return toolCalls;
}

function stripKimiK2(content: string): string {
  return stripBetweenTags(content, KIMI_OPEN, KIMI_CLOSE);
}

function detectDeepSeekV3(content: string): boolean {
  return (
    content.includes(DEEPSEEK_V3_OPEN) || content.includes(DEEPSEEK_V3_CLOSE)
  );
}

function parseDeepSeekV3(content: string): ParsedToolCall[] {
  const blocks = extractBetweenTags(
    content,
    DEEPSEEK_V3_OPEN,
    DEEPSEEK_V3_CLOSE,
  );
  const toolCalls: ParsedToolCall[] = [];

  for (const block of blocks) {
    const lines = block
      .split("\n")
      .map((line) => line.trim())
      .filter(Boolean);
    for (const line of lines) {
      toolCalls.push(...parseJsonToolCalls(line));
    }
  }

  return toolCalls;
}

function stripDeepSeekV3(content: string): string {
  return stripBetweenTags(content, DEEPSEEK_V3_OPEN, DEEPSEEK_V3_CLOSE);
}

function detectMistral(content: string): boolean {
  return content.includes(MISTRAL_MARKER);
}

function parseMistral(content: string): ParsedToolCall[] {
  const markerIndex = content.indexOf(MISTRAL_MARKER);
  if (markerIndex === -1) {
    return [];
  }

  const remainder = content.slice(markerIndex + MISTRAL_MARKER.length).trim();
  const segment = findFirstBalancedJsonSegment(remainder) ?? remainder;
  return parseJsonToolCalls(segment);
}

function stripMistral(content: string): string {
  const markerIndex = content.indexOf(MISTRAL_MARKER);
  if (markerIndex === -1) {
    return content;
  }

  const remainder = content.slice(markerIndex + MISTRAL_MARKER.length);
  const segment = findFirstBalancedJsonSegment(remainder);
  if (!segment) {
    return content.slice(0, markerIndex);
  }

  return content.replace(`${MISTRAL_MARKER}${segment}`, "");
}

function detectMiniMax(content: string): boolean {
  return content.includes("minimax:tool_call");
}

function parseMiniMax(content: string): ParsedToolCall[] {
  const blocks = extractBetweenTags(content, MINIMAX_OPEN, MINIMAX_CLOSE);
  return parseNamedXmlInvocations(blocks);
}

function stripMiniMax(content: string): string {
  return stripBetweenTags(content, MINIMAX_OPEN, MINIMAX_CLOSE);
}

function detectDSML(content: string): boolean {
  return content.includes("dsml:tool_call") || content.includes(DSML_ALT_OPEN);
}

function parseDSML(content: string): ParsedToolCall[] {
  const normalizedContent = normalizeDsmlPrefixedTags(content);
  const blocks = [
    ...extractBetweenTags(normalizedContent, DSML_OPEN, DSML_CLOSE),
    ...extractBetweenTags(normalizedContent, "<tool_calls>", "</tool_calls>"),
    ...extractBetweenTags(normalizedContent, "<tool_call>", "</tool_call>"),
  ];
  return parseNamedXmlInvocations(blocks);
}

function stripDSML(content: string): string {
  return stripBetweenTags(
    stripBetweenTags(content, DSML_OPEN, DSML_CLOSE),
    DSML_ALT_OPEN,
    DSML_ALT_CLOSE,
  );
}

function normalizeDsmlPrefixedTags(content: string): string {
  return content
    .replaceAll("<｜｜DSML｜｜", "<")
    .replaceAll("</｜｜DSML｜｜", "</");
}

function parseNamedXmlInvocations(blocks: string[]): ParsedToolCall[] {
  const toolCalls: ParsedToolCall[] = [];

  for (const block of blocks) {
    const invokeMatch = block.match(
      /<invoke\b[^>]*\bname="([^"]+)"[^>]*>([\s\S]*?)<\/invoke>/,
    );
    if (!invokeMatch) {
      continue;
    }

    const toolName = invokeMatch[1];
    const parameters = parseParameterTags(invokeMatch[2]);
    const jsonArguments = parseToolArguments(invokeMatch[2].trim());
    const argumentsObject = parameters ?? jsonArguments ?? {};

    toolCalls.push({
      toolCallId: generateToolCallId(toolCalls.length, toolName),
      toolName,
      arguments: argumentsObject,
    });
  }

  return toolCalls;
}

function detectXml(content: string): boolean {
  XML_TOOL_BLOCK_RE.lastIndex = 0;
  XML_FUNCTION_NAME_RE.lastIndex = 0;
  XML_FUNCTION_EQUALS_RE.lastIndex = 0;
  return (
    XML_TOOL_BLOCK_RE.test(content) ||
    XML_FUNCTION_NAME_RE.test(content) ||
    XML_FUNCTION_EQUALS_RE.test(content)
  );
}

function parseXml(content: string): ParsedToolCall[] {
  const toolCalls: ParsedToolCall[] = [];
  XML_TOOL_BLOCK_RE.lastIndex = 0;
  XML_FUNCTION_NAME_RE.lastIndex = 0;
  XML_FUNCTION_EQUALS_RE.lastIndex = 0;

  for (const match of content.matchAll(XML_TOOL_BLOCK_RE)) {
    const blockContent = match[2].trim();
    const parsed = parseJsonToolCalls(blockContent);
    if (parsed.length > 0) {
      toolCalls.push(...parsed);
      continue;
    }

    toolCalls.push(...parseXml(blockContent));
  }

  XML_TOOL_BLOCK_RE.lastIndex = 0;
  const wrapperStrippedContent = content.replace(XML_TOOL_BLOCK_RE, "");

  for (const match of wrapperStrippedContent.matchAll(XML_FUNCTION_NAME_RE)) {
    const toolName = match[1].trim();
    const body = match[2].trim();
    const parameters = parseParameterTags(body);
    const jsonArguments = parseToolArguments(body);
    const argumentsObject = parameters ?? jsonArguments ?? {};

    toolCalls.push({
      toolCallId: generateToolCallId(toolCalls.length, toolName),
      toolName,
      arguments: argumentsObject,
    });
  }

  for (const match of wrapperStrippedContent.matchAll(XML_FUNCTION_EQUALS_RE)) {
    const toolName = match[1].trim();
    const body = match[2].trim();
    const parameters = parseParameterTags(body);
    const jsonArguments = parseToolArguments(body);
    const argumentsObject = parameters ?? jsonArguments ?? {};

    toolCalls.push({
      toolCallId: generateToolCallId(toolCalls.length, toolName),
      toolName,
      arguments: argumentsObject,
    });
  }

  return toolCalls;
}

function stripXml(content: string): string {
  let stripped = content;

  for (const tagName of [
    "tool_call",
    "tool_calls",
    "tool_result",
    "function_call",
    "function_calls",
  ]) {
    stripped = stripWithRegex(
      stripped,
      new RegExp(`<${tagName}\\b[^>]*>[\\s\\S]*?<\\/${tagName}>`, "gi"),
    );
  }

  XML_FUNCTION_NAME_RE.lastIndex = 0;
  stripped = stripWithRegex(stripped, XML_FUNCTION_NAME_RE);

  XML_FUNCTION_EQUALS_RE.lastIndex = 0;
  stripped = stripWithRegex(stripped, XML_FUNCTION_EQUALS_RE);

  // Also strip <function=name> format used by some models
  stripped = stripped.replace(
    /<function\s*=\s*\w+\b[^>]*>[\s\S]*?<\/function>/gi,
    " ",
  );

  // Collapse multiple spaces into single space (but preserve leading/trailing)
  stripped = stripped.replace(/ {2,}/g, " ");

  return stripped;
}

function normalizeFallbackFormat(
  format?: FeatherlessCompatibleToolCallFallbackMode,
): FeatherlessCompatibleToolCallFormat | undefined {
  if (!format || format === "auto" || format === "disabled") {
    return undefined;
  }

  return format === "hermes" ? "fenced-json" : format;
}

const FORMATS: FormatHandler[] = [
  {
    name: "fenced-json",
    detect: detectFencedJson,
    parse: parseFencedJson,
    strip: stripFencedJson,
  },
  {
    name: "gemma4",
    detect: detectGemma4,
    parse: parseGemma4,
    strip: stripGemma4,
  },
  {
    name: "kimi-k2",
    detect: detectKimiK2,
    parse: parseKimiK2,
    strip: stripKimiK2,
  },
  {
    name: "deepseek-v3",
    detect: detectDeepSeekV3,
    parse: parseDeepSeekV3,
    strip: stripDeepSeekV3,
  },
  {
    name: "mistral",
    detect: detectMistral,
    parse: parseMistral,
    strip: stripMistral,
  },
  {
    name: "minimax",
    detect: detectMiniMax,
    parse: parseMiniMax,
    strip: stripMiniMax,
  },
  { name: "dsml", detect: detectDSML, parse: parseDSML, strip: stripDSML },
  { name: "xml", detect: detectXml, parse: parseXml, strip: stripXml },
];

/**
 * Detect and parse tool calls from raw text content.
 *
 * Tries each registered format in order. Returns the first format
 * that successfully detects markers and parses tool calls.
 */
export function detectAndParseToolCalls(
  content: string,
  forceFormat?: FeatherlessCompatibleToolCallFallbackMode,
): ToolCallDetection {
  if (!content || typeof content !== "string") {
    return { format: null, toolCalls: [], cleanedContent: content ?? "" };
  }

  const reasoningContent = extractReasoningBlocks(content);

  if (forceFormat) {
    const handler = FORMATS.find(
      (format) => format.name === normalizeFallbackFormat(forceFormat),
    );
    if (!handler?.detect(content)) {
      return { format: null, toolCalls: [], cleanedContent: content };
    }

    const toolCalls = reindexToolCalls(handler.parse(content));
    if (toolCalls.length === 0) {
      return { format: null, toolCalls: [], cleanedContent: content };
    }

    return {
      format: handler.name,
      toolCalls,
      cleanedContent: stripReasoningBlocks(handler.strip(content)),
      ...(reasoningContent ? { reasoningContent } : {}),
    };
  }

  const allToolCalls: ParsedToolCall[] = [];
  let detectedFormat: FeatherlessCompatibleToolCallFormat | null = null;
  let cleanedContent = content;

  for (const handler of FORMATS) {
    if (!handler.detect(content)) {
      continue;
    }

    const parsed = handler.parse(content);
    if (parsed.length === 0) {
      continue;
    }

    if (!detectedFormat) {
      detectedFormat = handler.name;
    }

    allToolCalls.push(...parsed);
    cleanedContent = handler.strip(cleanedContent);
  }

  if (allToolCalls.length === 0) {
    return {
      format: null,
      toolCalls: [],
      cleanedContent: stripReasoningBlocks(content),
      ...(reasoningContent ? { reasoningContent } : {}),
    };
  }

  return {
    format: detectedFormat,
    toolCalls: reindexToolCalls(allToolCalls),
    cleanedContent: stripReasoningBlocks(cleanedContent),
    ...(reasoningContent ? { reasoningContent } : {}),
  };
}

export function stripToolCallMarkers(
  content: string,
  format: FeatherlessCompatibleToolCallFallbackMode,
): string {
  const handler = FORMATS.find(
    (entry) => entry.name === normalizeFallbackFormat(format),
  );
  return handler ? handler.strip(content) : content;
}
