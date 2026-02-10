"use client";

import { useCallback, useRef, useState } from "react";
import { API_BASE } from "@/lib/api";

interface SSEEvent {
  type: string;
  content?: unknown;
  data?: string;
  node?: string;
  [key: string]: unknown;
}

export function useSSE() {
  const [events, setEvents] = useState<SSEEvent[]>([]);
  const [isStreaming, setIsStreaming] = useState(false);
  const abortRef = useRef<AbortController | null>(null);

  const startStream = useCallback(
    async (path: string, body: unknown, onEvent?: (event: SSEEvent) => void) => {
      setEvents([]);
      setIsStreaming(true);

      const token = localStorage.getItem("access_token");
      const headers: Record<string, string> = { "Content-Type": "application/json" };
      if (token) headers["Authorization"] = `Bearer ${token}`;

      abortRef.current = new AbortController();

      try {
        const res = await fetch(`${API_BASE}${path}`, {
          method: "POST",
          headers,
          body: JSON.stringify(body),
          signal: abortRef.current.signal,
        });

        const reader = res.body?.getReader();
        if (!reader) return;

        const decoder = new TextDecoder();
        let buffer = "";

        while (true) {
          const { value, done } = await reader.read();
          if (done) break;

          buffer += decoder.decode(value, { stream: true });
          const lines = buffer.split("\n");
          buffer = lines.pop() || "";

          for (const line of lines) {
            if (!line.startsWith("data: ")) continue;
            try {
              const event: SSEEvent = JSON.parse(line.substring(6));
              setEvents((prev) => [...prev, event]);
              onEvent?.(event);
            } catch {
              // skip malformed
            }
          }
        }
      } catch (e) {
        if ((e as Error).name !== "AbortError") {
          setEvents((prev) => [
            ...prev,
            { type: "error", content: (e as Error).message },
          ]);
        }
      } finally {
        setIsStreaming(false);
      }
    },
    []
  );

  const stopStream = useCallback(() => {
    abortRef.current?.abort();
    setIsStreaming(false);
  }, []);

  return { events, isStreaming, startStream, stopStream };
}
