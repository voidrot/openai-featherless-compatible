import { describe, it, expect, vi, beforeEach } from "vitest";
import { createFeatherlessCompatibleProvider } from "../src/provider";
import type { LanguageModelV3 } from "@ai-sdk/provider";

// Mock dependencies
vi.mock("@ai-sdk/openai-compatible", () => ({
  createOpenAICompatible: vi.fn((options: any) => ({
    name: options.name,
    languageModel: vi.fn((modelId: string) => ({
      specificationVersion: "v3",
      provider: "openai",
      modelId,
      supportedUrls: {},
      doGenerate: vi.fn(),
      doStream: vi.fn(),
    })),
    chatModel: vi.fn((modelId: string) => ({
      specificationVersion: "v3",
      provider: "openai",
      modelId,
      supportedUrls: {},
      doGenerate: vi.fn(),
      doStream: vi.fn(),
    })),
    completionModel: vi.fn((modelId: string) => ({
      specificationVersion: "v3",
      provider: "openai",
      modelId,
      supportedUrls: {},
      doGenerate: vi.fn(),
      doStream: vi.fn(),
    })),
    embeddingModel: vi.fn((modelId: string) => ({
      specificationVersion: "v3",
      provider: "openai",
      modelId,
      doEmbed: vi.fn(),
    })),
    textEmbeddingModel: vi.fn((modelId: string) => ({
      specificationVersion: "v3",
      provider: "openai",
      modelId,
      doEmbed: vi.fn(),
    })),
    imageModel: vi.fn((modelId: string) => ({
      specificationVersion: "v3",
      provider: "openai",
      modelId,
      doGenerate: vi.fn(),
    })),
  })),
}));

describe("featherless-compatible-provider", () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  describe("createFeatherlessCompatibleProvider", () => {
    it("should create provider with default settings", () => {
      const provider = createFeatherlessCompatibleProvider({
        name: "test-provider",
        baseURL: "https://api.test.com/v1",
        apiKey: "test-key",
      });

      expect(provider).toBeDefined();
      expect(provider.toolCallFallback).toBe("auto");
      expect(provider.specificationVersion).toBe("v3");
      expect(typeof provider.languageModel).toBe("function");
      expect(typeof provider.embeddingModel).toBe("function");
      expect(typeof provider.imageModel).toBe("function");
    });

    it("should create provider with custom toolCallFallback", () => {
      const provider = createFeatherlessCompatibleProvider({
        name: "test-provider",
        baseURL: "https://api.test.com/v1",
        apiKey: "test-key",
        toolCallFallback: "fenced-json",
      });

      expect(provider.toolCallFallback).toBe("fenced-json");
    });

    it("should create provider with custom generateToolCallId", () => {
      const customId = (index: number, toolName: string) =>
        `custom_${index}_${toolName}`;
      const provider = createFeatherlessCompatibleProvider({
        name: "test-provider",
        baseURL: "https://api.test.com/v1",
        apiKey: "test-key",
        generateToolCallId: customId,
      });

      expect(provider.toolCallFallback).toBe("auto");
    });

    it("should create model using languageModel method", () => {
      const provider = createFeatherlessCompatibleProvider({
        name: "test-provider",
        baseURL: "https://api.test.com/v1",
        apiKey: "test-key",
      });

      const model = provider.languageModel("test-model");
      expect(model).toBeDefined();
      expect(model.modelId).toBe("test-model");
    });

    it("should expose the callable provider and chat aliases", () => {
      const provider = createFeatherlessCompatibleProvider({
        name: "test-provider",
        baseURL: "https://api.test.com/v1",
        apiKey: "test-key",
      });

      const callableModel = provider("test-model");
      const chatModel = provider.chatModel("test-model");
      const completionModel = provider.completionModel("test-model");

      expect(callableModel.modelId).toBe("test-model");
      expect(chatModel.modelId).toBe("test-model");
      expect(completionModel.modelId).toBe("test-model");
      expect(callableModel.provider).toContain("test-provider");
      expect(chatModel.provider).toContain("test-provider");
      expect(completionModel.provider).toContain("test-provider");
    });

    it("should delegate embeddingModel to inner provider", () => {
      const provider = createFeatherlessCompatibleProvider({
        name: "test-provider",
        baseURL: "https://api.test.com/v1",
        apiKey: "test-key",
      });

      const embeddingModel = provider.embeddingModel("embedding-model");
      expect(embeddingModel).toBeDefined();
    });

    it("should expose textEmbeddingModel alias", () => {
      const provider = createFeatherlessCompatibleProvider({
        name: "test-provider",
        baseURL: "https://api.test.com/v1",
        apiKey: "test-key",
      });

      const embeddingModel = provider.textEmbeddingModel("embedding-model");
      expect(embeddingModel).toBeDefined();
    });

    it("should delegate imageModel to inner provider", () => {
      const provider = createFeatherlessCompatibleProvider({
        name: "test-provider",
        baseURL: "https://api.test.com/v1",
        apiKey: "test-key",
      });

      const imageModel = provider.imageModel("image-model");
      expect(imageModel).toBeDefined();
    });

    it("should pass through all provider options to inner provider", () => {
      const provider = createFeatherlessCompatibleProvider({
        name: "test-provider",
        baseURL: "https://api.test.com/v1",
        apiKey: "test-key",
        headers: { "X-Custom": "value" },
      });

      expect(provider).toBeDefined();
    });
  });

  describe("FeatherlessCompatibleProvider interface", () => {
    it("should implement the provider v3 surface", () => {
      const provider = createFeatherlessCompatibleProvider({
        name: "test-provider",
        baseURL: "https://api.test.com/v1",
        apiKey: "test-key",
      });

      const model = provider.languageModel("test-model");
      expect(model).toBeInstanceOf(Object);
      expect(model).toHaveProperty("doGenerate");
      expect(model).toHaveProperty("doStream");
      expect(model).toHaveProperty("supportedUrls");
    });

    it("should have correct specificationVersion", () => {
      const provider = createFeatherlessCompatibleProvider({
        name: "test-provider",
        baseURL: "https://api.test.com/v1",
        apiKey: "test-key",
      });

      expect(provider.specificationVersion).toBe("v3");
    });

    it("should have correct toolCallFallback property", () => {
      const provider = createFeatherlessCompatibleProvider({
        name: "test-provider",
        baseURL: "https://api.test.com/v1",
        apiKey: "test-key",
        toolCallFallback: "gemma4",
      });

      expect(provider.toolCallFallback).toBe("gemma4");
    });
  });

  describe("Integration scenarios", () => {
    it("should create a complete provider chain", () => {
      const provider = createFeatherlessCompatibleProvider({
        name: "my-featherless",
        baseURL: "https://api.example.com/v1",
        apiKey: process.env.API_KEY || "test-key",
        toolCallFallback: "auto",
      });

      const model = provider.languageModel("featherless-3-7b");
      expect(model).toBeDefined();
      expect(model.modelId).toBe("featherless-3-7b");
      expect(model.provider).toContain("my-featherless");
    });

    it("should work with different fallback modes", () => {
      const autoProvider = createFeatherlessCompatibleProvider({
        name: "test-auto",
        baseURL: "https://api.test.com/v1",
        apiKey: "test-key",
        toolCallFallback: "auto",
      });

      const featherlessProvider = createFeatherlessCompatibleProvider({
        name: "test-featherless",
        baseURL: "https://api.test.com/v1",
        apiKey: "test-key",
        toolCallFallback: "fenced-json",
      });

      const gemmaProvider = createFeatherlessCompatibleProvider({
        name: "test-gemma",
        baseURL: "https://api.test.com/v1",
        apiKey: "test-key",
        toolCallFallback: "gemma4",
      });

      const xmlProvider = createFeatherlessCompatibleProvider({
        name: "test-xml",
        baseURL: "https://api.test.com/v1",
        apiKey: "test-key",
        toolCallFallback: "xml",
      });

      expect(autoProvider.toolCallFallback).toBe("auto");
      expect(featherlessProvider.toolCallFallback).toBe("fenced-json");
      expect(gemmaProvider.toolCallFallback).toBe("gemma4");
      expect(xmlProvider.toolCallFallback).toBe("xml");
    });
  });
});
