import { describe, expect, it } from "vitest";
import {
  buildToolSchemaMap,
  coerceParsedToolCalls,
  detectAndParseToolCalls,
  generateToolCallId,
  stripToolCallMarkers,
} from "../src/tool-call-parser";

describe("featherless-compatible-tool-call-parser", () => {
  describe("generateToolCallId", () => {
    it("should generate unique IDs with index and tool name", () => {
      expect(generateToolCallId(0, "search")).toBe("tc_0_search");
      expect(generateToolCallId(1, "search")).toBe("tc_1_search");
      expect(generateToolCallId(0, "get_weather")).toBe("tc_0_get_weather");
    });
  });

  describe("Fenced JSON Format", () => {
    it("should detect fenced JSON format with backtick blocks", () => {
      const content =
        'Here is the result:\n```\n{"name": "search", "arguments": {"query": "test"}}\n```';
      const result = detectAndParseToolCalls(content);
      expect(result.format).toBe("fenced-json");
      expect(result.toolCalls).toHaveLength(1);
      expect(result.toolCalls[0].toolName).toBe("search");
      expect(result.toolCalls[0].arguments).toEqual({ query: "test" });
    });

    it("should detect fenced JSON format with json blocks", () => {
      const content =
        'Here is the result:\n```json\n{"name": "get_weather", "arguments": {"location": "NYC"}}\n```';
      const result = detectAndParseToolCalls(content);
      expect(result.format).toBe("fenced-json");
      expect(result.toolCalls).toHaveLength(1);
      expect(result.toolCalls[0].toolName).toBe("get_weather");
    });

    it("should strip fenced JSON markers", () => {
      const content =
        'Result:\n```\n{"name": "search", "arguments": {}}\n```\nDone';
      const result = stripToolCallMarkers(content, "fenced-json");
      expect(result).toBe("Result:\n\nDone");
    });

    it("should handle multiple tool calls in fenced JSON format", () => {
      const content =
        '```\n{"name": "search", "arguments": {"query": "test1"}}\n```\n```\n{"name": "search", "arguments": {"query": "test2"}}\n```';
      const result = detectAndParseToolCalls(content);
      expect(result.toolCalls).toHaveLength(2);
      expect(result.toolCalls[0].toolName).toBe("search");
      expect(result.toolCalls[1].toolName).toBe("search");
    });

    it("should treat hermes as a deprecated alias for fenced-json", () => {
      const content =
        '```\n{"name": "search", "arguments": {"query": "test"}}\n```';
      const result = detectAndParseToolCalls(content, "hermes");
      expect(result.format).toBe("fenced-json");
      expect(result.toolCalls).toHaveLength(1);
    });
  });

  describe("Gemma 4 Format", () => {
    it("should detect Gemma 4 format", () => {
      const content =
        'Result:<|tool_call>{"name": "search", "arguments": {"query": "test"}}<|end_of_tool_call|>';
      const result = detectAndParseToolCalls(content);
      expect(result.format).toBe("gemma4");
      expect(result.toolCalls).toHaveLength(1);
      expect(result.toolCalls[0].toolName).toBe("search");
    });

    it("should parse Gemma 4 with paren syntax", () => {
      const content =
        '<|tool_call>call:search(query="test")<|end_of_tool_call|>';
      const result = detectAndParseToolCalls(content);
      expect(result.toolCalls).toHaveLength(1);
      expect(result.toolCalls[0].toolName).toBe("search");
      expect(result.toolCalls[0].arguments).toEqual({ query: "test" });
    });

    it("should strip Gemma 4 markers", () => {
      const content = '<|tool_call>{"name": "search"}<|end_of_tool_call|>Done';
      const result = stripToolCallMarkers(content, "gemma4");
      expect(result).toBe("Done");
    });

    it("should handle multiple Gemma 4 tool calls", () => {
      const content =
        '<|tool_call>{"name": "search1", "arguments": {}}<|end_of_tool_call|><|tool_call>{"name": "search2", "arguments": {}}<|end_of_tool_call|>';
      const result = detectAndParseToolCalls(content);
      expect(result.toolCalls).toHaveLength(2);
    });
  });

  describe("Kimi K2 Format", () => {
    it("should detect Kimi K2 format", () => {
      const content =
        '<|tool_calls_section_begin|>\n{"name": "search", "arguments": {}}\n<|tool_calls_section_end|>';
      const result = detectAndParseToolCalls(content);
      expect(result.format).toBe("kimi-k2");
      expect(result.toolCalls).toHaveLength(1);
    });

    it("should strip Kimi K2 markers", () => {
      const content =
        '<|tool_calls_section_begin|>\n{"name": "search"}\n<|tool_calls_section_end|>Done';
      const result = stripToolCallMarkers(content, "kimi-k2");
      expect(result).toBe("Done");
    });
  });

  describe("DeepSeek V3 Format", () => {
    it("should detect DeepSeek V3 format", () => {
      const content =
        '<｜tool▁calls▁begin｜>\n{"name": "search", "arguments": {}}\n<｜tool▁calls▁end｜>';
      const result = detectAndParseToolCalls(content);
      expect(result.format).toBe("deepseek-v3");
      expect(result.toolCalls).toHaveLength(1);
    });

    it("should strip DeepSeek V3 markers", () => {
      const content =
        '<｜tool▁calls▁begin｜>\n{"name": "search"}\n<｜tool▁calls▁end｜>Done';
      const result = stripToolCallMarkers(content, "deepseek-v3");
      expect(result).toBe("Done");
    });
  });

  describe("Mistral Format", () => {
    it("should detect Mistral format", () => {
      const content =
        'Result:\n[TOOL_CALLS][{"name": "search", "arguments": {}}]';
      const result = detectAndParseToolCalls(content);
      expect(result.format).toBe("mistral");
      expect(result.toolCalls).toHaveLength(1);
    });

    it("should strip Mistral markers", () => {
      const content = '[TOOL_CALLS][{"name": "search"}]Done';
      const result = stripToolCallMarkers(content, "mistral");
      expect(result).toBe("Done");
    });
  });

  describe("MiniMax Format", () => {
    it("should detect MiniMax format", () => {
      const content =
        '<minimax:tool_call><invoke name="search"><parameter name="query">test</parameter></invoke></minimax:tool_call>';
      const result = detectAndParseToolCalls(content);
      expect(result.format).toBe("minimax");
      expect(result.toolCalls).toHaveLength(1);
      expect(result.toolCalls[0].toolName).toBe("search");
      expect(result.toolCalls[0].arguments).toEqual({ query: "test" });
    });

    it("should strip MiniMax markers", () => {
      const content =
        '<minimax:tool_call><invoke name="search"></invoke></minimax:tool_call>Done';
      const result = stripToolCallMarkers(content, "minimax");
      expect(result).toBe("Done");
    });
  });

  describe("DeepSeek V4 (DSML) Format", () => {
    it("should detect DSML format", () => {
      const content =
        '<dsml:tool_call><invoke name="search"><parameter name="query">test</parameter></invoke></dsml:tool_call>';
      const result = detectAndParseToolCalls(content);
      expect(result.format).toBe("dsml");
      expect(result.toolCalls).toHaveLength(1);
    });

    it("should detect DSML-prefixed tool_calls blocks", () => {
      const content =
        '<｜｜DSML｜｜tool_calls>\n<｜｜DSML｜｜invoke name="todowrite">\n<｜｜DSML｜｜parameter name="todos" string="false">{"priority":"high","content":"Map omnisync codebase structure and entry points","status":"completed"},{"priority":"high","content":"Trace device sync logic from legacy MySQL","status":"completed"},{"priority":"high","content":"Investigate device 866833045389613 in legacy MySQL","status":"completed"},{"priority":"high","content":"Identify root cause of sync failure - scheduler stopped after Apr 10","status":"in_progress"},{"priority":"medium","content":"Investigate why scheduler stopped running","status":"pending"}</｜｜DSML｜｜parameter>\n</｜｜DSML｜｜invoke>\n</｜｜DSML｜｜tool_calls>';

      const result = detectAndParseToolCalls(content);

      expect(result.format).toBe("dsml");
      expect(result.toolCalls).toHaveLength(1);
      expect(result.toolCalls[0].toolName).toBe("todowrite");
      expect(result.toolCalls[0].arguments).toEqual({
        todos: {
          priority: "high",
          content: "Map omnisync codebase structure and entry points",
          status: "completed",
        },
      });
    });

    it("should strip DSML markers", () => {
      const content =
        '<dsml:tool_call><invoke name="search"></invoke></dsml:tool_call>Done';
      const result = stripToolCallMarkers(content, "dsml");
      expect(result).toBe("Done");

      const dsmlPrefixedContent =
        '<｜｜DSML｜｜tool_calls><｜｜DSML｜｜invoke name="search"></｜｜DSML｜｜invoke></｜｜DSML｜｜tool_calls>Done';
      const dsmlPrefixedResult = stripToolCallMarkers(
        dsmlPrefixedContent,
        "dsml",
      );
      expect(dsmlPrefixedResult).toBe("Done");
    });
  });

  describe("Generic XML Format", () => {
    it("should detect tool_call XML blocks", () => {
      const content =
        '<tool_call>{"name": "search", "arguments": {"query": "test"}}</tool_call>';
      const result = detectAndParseToolCalls(content);
      expect(result.format).toBe("xml");
      expect(result.toolCalls).toHaveLength(1);
      expect(result.toolCalls[0].toolName).toBe("search");
      expect(result.toolCalls[0].arguments).toEqual({ query: "test" });
    });

    it("should detect function name XML blocks", () => {
      const content =
        '<function name="read_file">{"path":"/tmp/demo.txt"}</function>';
      const result = detectAndParseToolCalls(content);
      expect(result.format).toBe("xml");
      expect(result.toolCalls).toHaveLength(1);
      expect(result.toolCalls[0].toolName).toBe("read_file");
      expect(result.toolCalls[0].arguments).toEqual({ path: "/tmp/demo.txt" });
    });

    it("should detect StepFun-style function equals syntax with parameter equals tags", () => {
      const content = `<tool_call>
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
</tool_call>`;

      const result = detectAndParseToolCalls(content);

      expect(result.format).toBe("xml");
      expect(result.toolCalls).toHaveLength(1);
      expect(result.toolCalls[0].toolName).toBe("grep");
      expect(result.toolCalls[0].arguments).toEqual({
        "-n": true,
        output: "content",
        path: "/home/buck/Projects/homelab/argocd-apps/apps/ai-tools/litellm/litellm.yaml",
        pattern: "^          search_tools:",
      });
    });

    it("should detect function_calls JSON arrays", () => {
      const content =
        '<function_calls>[{"name":"search","arguments":{"query":"one"}},{"name":"search","arguments":{"query":"two"}}]</function_calls>';
      const result = detectAndParseToolCalls(content);
      expect(result.format).toBe("xml");
      expect(result.toolCalls).toHaveLength(2);
    });

    it("should strip XML markers", () => {
      const content = '<tool_call>{"name": "search"}</tool_call>Done';
      const result = stripToolCallMarkers(content, "xml");
      expect(result).toBe("Done");
    });

    it("should strip reasoning tags from cleaned content when tool calls are parsed", () => {
      const content =
        '<think>internal reasoning</think>Visible <tool_call>{"name":"search","arguments":{}}</tool_call>';
      const result = detectAndParseToolCalls(content);
      expect(result.cleanedContent).toBe("Visible ");
      expect(result.reasoningContent).toBe("internal reasoning");
    });

    it("should strip tool-call markers from reasoning blocks", () => {
      const content =
        "<think>Let me search first <function=search></function> then continue thinking</think>Final answer";
      const result = detectAndParseToolCalls(content);
      expect(result.reasoningContent).toBe(
        "Let me search first then continue thinking",
      );
      expect(result.cleanedContent).toBe("Final answer");
    });
  });

  describe("detectAndParseToolCalls", () => {
    it("should return empty result for empty content", () => {
      const result = detectAndParseToolCalls("");
      expect(result).toEqual({
        format: null,
        toolCalls: [],
        cleanedContent: "",
      });
    });

    it("should return empty result for null content", () => {
      const result = detectAndParseToolCalls(null as unknown as string);
      expect(result).toEqual({
        format: null,
        toolCalls: [],
        cleanedContent: "",
      });
    });

    it("should auto-detect format when not specified", () => {
      const content = '```\n{"name": "search", "arguments": {}}\n```';
      const result = detectAndParseToolCalls(content);
      expect(result.format).toBe("fenced-json");
    });

    it("should force specific format when specified", () => {
      const content =
        '<|tool_call>{"name": "search", "arguments": {}}<|end_of_tool_call|>';
      const result = detectAndParseToolCalls(content, "gemma4");
      expect(result.format).toBe("gemma4");
    });

    it("should return null format when no tool calls detected", () => {
      const content = "No tool calls here";
      const result = detectAndParseToolCalls(content);
      expect(result.format).toBeNull();
      expect(result.toolCalls).toHaveLength(0);
    });

    it("should handle malformed JSON gracefully", () => {
      const content = '```\n{"name": "search", "arguments": invalid}\n```';
      const result = detectAndParseToolCalls(content);
      expect(result.toolCalls).toHaveLength(0);
    });

    it("should recover common loose JSON variants", () => {
      const content =
        '```json\n{"name": "search", "arguments": {"enabled": True, "limit": 3,}}\n```';
      const result = detectAndParseToolCalls(content);
      expect(result.toolCalls).toHaveLength(1);
      expect(result.toolCalls[0].arguments).toEqual({
        enabled: true,
        limit: 3,
      });
    });

    it("should handle multiple formats in one content", () => {
      const content =
        '```\n{"name": "search1", "arguments": {}}\n```\n<|tool_call>{"name": "search2", "arguments": {}}<|end_of_tool_call|>';
      const result = detectAndParseToolCalls(content);
      expect(result.toolCalls).toHaveLength(2);
    });

    it("should clean content by removing tool call markers", () => {
      const content =
        'Result:\n```\n{"name": "search", "arguments": {}}\n```\nDone';
      const result = detectAndParseToolCalls(content);
      expect(result.cleanedContent).not.toContain("```");
      expect(result.cleanedContent).toContain("Result:");
      expect(result.cleanedContent).toContain("Done");
    });
  });

  describe("stripToolCallMarkers", () => {
    it("should strip markers for valid format", () => {
      const content = '<|tool_call>{"name": "search"}<|end_of_tool_call|>';
      const result = stripToolCallMarkers(content, "gemma4");
      expect(result).toBe("");
    });

    it("should return original content for invalid format", () => {
      const content = '<|tool_call>{"name": "search"}<|end_of_tool_call|>';
      const result = stripToolCallMarkers(content, "invalid");
      expect(result).toBe(content);
    });

    it("should handle multiple markers", () => {
      const content =
        '<|tool_call>{"name": "search1"}<|end_of_tool_call|><|tool_call>{"name": "search2"}<|end_of_tool_call|>';
      const result = stripToolCallMarkers(content, "gemma4");
      expect(result).toBe("");
    });
  });

  describe("Edge Cases", () => {
    it("should handle empty arguments object", () => {
      const content = '```\n{"name": "search", "arguments": {}}\n```';
      const result = detectAndParseToolCalls(content);
      expect(result.toolCalls[0].arguments).toEqual({});
    });

    it("should handle nested arguments", () => {
      const content =
        '```\n{"name": "search", "arguments": {"filters": {"query": "test", "limit": 10}}}\n```';
      const result = detectAndParseToolCalls(content);
      expect(result.toolCalls[0].arguments).toEqual({
        filters: { query: "test", limit: 10 },
      });
    });

    it("should handle string arguments", () => {
      const content =
        '```\n{"name": "search", "arguments": {"query": "hello world"}}\n```';
      const result = detectAndParseToolCalls(content);
      expect(result.toolCalls[0].arguments).toEqual({ query: "hello world" });
    });

    it("should handle boolean arguments", () => {
      const content =
        '```\n{"name": "search", "arguments": {"include_deleted": false}}\n```';
      const result = detectAndParseToolCalls(content);
      expect(result.toolCalls[0].arguments).toEqual({ include_deleted: false });
    });

    it("should handle number arguments", () => {
      const content =
        '```\n{"name": "search", "arguments": {"limit": 42}}\n```';
      const result = detectAndParseToolCalls(content);
      expect(result.toolCalls[0].arguments).toEqual({ limit: 42 });
    });

    it("should handle null arguments", () => {
      const content =
        '```\n{"name": "search", "arguments": {"retry": null}}\n```';
      const result = detectAndParseToolCalls(content);
      expect(result.toolCalls[0].arguments).toEqual({ retry: null });
    });

    it("should handle array arguments", () => {
      const content =
        '```\n{"name": "search", "arguments": {"tags": ["a", "b", "c"]}}\n```';
      const result = detectAndParseToolCalls(content);
      expect(result.toolCalls[0].arguments).toEqual({ tags: ["a", "b", "c"] });
    });
  });

  describe("Argument coercion", () => {
    it("should coerce values to the declared input schema", () => {
      const detected = detectAndParseToolCalls(
        '```json\n{"name":"search","arguments":{"tags":"a","limit":"42","exact":"true"}}\n```',
      );
      const toolSchemas = buildToolSchemaMap([
        {
          type: "function" as const,
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
      ]);

      const coerced = coerceParsedToolCalls(detected.toolCalls, toolSchemas);
      expect(coerced[0].arguments).toEqual({
        tags: ["a"],
        limit: 42,
        exact: true,
      });
    });
  });
});
