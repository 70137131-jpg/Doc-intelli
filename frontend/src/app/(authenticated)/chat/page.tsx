"use client";

import { useState, useRef, useEffect, useCallback } from "react";
import {
  Send,
  Plus,
  MessageSquare,
  Loader2,
  FileText,
  Trash2,
} from "lucide-react";
import { Card, CardContent } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Textarea } from "@/components/ui/textarea";
import { ScrollArea } from "@/components/ui/scroll-area";
import { Separator } from "@/components/ui/separator";
import { Badge } from "@/components/ui/badge";
import { Skeleton } from "@/components/ui/skeleton";
import { api } from "@/lib/api";
import { useSSE } from "@/hooks/use-sse";
import type { Conversation, Message, Source } from "@/types";

export default function ChatPage() {
  const [conversations, setConversations] = useState<Conversation[]>([]);
  const [activeConvoId, setActiveConvoId] = useState<string | null>(null);
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState("");
  const [sending, setSending] = useState(false);
  const [loadingConvos, setLoadingConvos] = useState(true);
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const { isStreaming, startStream, stopStream } = useSSE();

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  // Load conversations
  const loadConversations = useCallback(async () => {
    try {
      const res = await api.get<{ conversations: Conversation[] }>("/chat/conversations");
      setConversations(res.conversations || []);
    } catch {
      // API unavailable
    } finally {
      setLoadingConvos(false);
    }
  }, []);

  useEffect(() => {
    loadConversations();
  }, [loadConversations]);

  // Load messages for active conversation
  useEffect(() => {
    if (!activeConvoId) {
      setMessages([]);
      return;
    }
    async function load() {
      try {
        const res = await api.get<{ messages: Message[] }>(
          `/chat/conversations/${activeConvoId}/messages`
        );
        setMessages(res.messages || []);
      } catch {
        setMessages([]);
      }
    }
    load();
  }, [activeConvoId]);

  const createNewConversation = async () => {
    try {
      const res = await api.post<Conversation>("/chat/conversations", {
        title: "New Chat",
      });
      setConversations((prev) => [res, ...prev]);
      setActiveConvoId(res.id);
      setMessages([]);
    } catch {
      // Fallback: just start a new local chat
      setActiveConvoId(null);
      setMessages([]);
    }
  };

  const deleteConversation = async (id: string) => {
    try {
      await api.delete(`/chat/conversations/${id}`);
    } catch {
      // Ignore errors
    }
    setConversations((prev) => prev.filter((c) => c.id !== id));
    if (activeConvoId === id) {
      setActiveConvoId(null);
      setMessages([]);
    }
  };

  const handleSend = async () => {
    if (!input.trim() || sending) return;

    const userMessage: Message = {
      id: crypto.randomUUID(),
      role: "user",
      content: input.trim(),
      created_at: new Date().toISOString(),
    };

    setMessages((prev) => [...prev, userMessage]);
    setInput("");
    setSending(true);

    // Add placeholder for assistant response
    const assistantId = crypto.randomUUID();
    const assistantMsg: Message = {
      id: assistantId,
      role: "assistant",
      content: "",
      created_at: new Date().toISOString(),
      sources: [],
    };
    setMessages((prev) => [...prev, assistantMsg]);

    try {
      // Try SSE streaming first
      let fullContent = "";
      const sources: Source[] = [];

      await startStream(
        "/chat/stream",
        {
          query: userMessage.content,
          conversation_id: activeConvoId,
        },
        (event) => {
          if (event.type === "token" || event.type === "chunk") {
            fullContent += event.data;
            setMessages((prev) =>
              prev.map((m) =>
                m.id === assistantId ? { ...m, content: fullContent } : m
              )
            );
          } else if (event.type === "sources") {
            try {
              const parsed = JSON.parse(event.data);
              sources.push(...(parsed.sources || []));
              setMessages((prev) =>
                prev.map((m) =>
                  m.id === assistantId ? { ...m, sources } : m
                )
              );
            } catch {
              // Ignore parse errors
            }
          } else if (event.type === "done") {
            // Final
          }
        }
      );

      // If streaming didn't produce output, fallback to regular POST
      if (!fullContent) {
        const res = await api.post<{ answer: string; sources?: Source[] }>("/chat/query", {
          query: userMessage.content,
          conversation_id: activeConvoId,
        });
        setMessages((prev) =>
          prev.map((m) =>
            m.id === assistantId
              ? { ...m, content: res.answer, sources: res.sources }
              : m
          )
        );
      }
    } catch {
      setMessages((prev) =>
        prev.map((m) =>
          m.id === assistantId
            ? { ...m, content: "Sorry, I could not process your request. Please try again." }
            : m
        )
      );
    } finally {
      setSending(false);
    }
  };

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  };

  return (
    <div className="flex h-[calc(100vh-7rem)] gap-4">
      {/* Sidebar - conversations */}
      <Card className="w-64 shrink-0 flex flex-col">
        <div className="p-3 border-b">
          <Button onClick={createNewConversation} className="w-full" size="sm">
            <Plus className="mr-2 h-4 w-4" />
            New Chat
          </Button>
        </div>
        <ScrollArea className="flex-1">
          <div className="p-2 space-y-1">
            {loadingConvos ? (
              [...Array(3)].map((_, i) => <Skeleton key={i} className="h-10 w-full" />)
            ) : conversations.length === 0 ? (
              <p className="text-xs text-muted-foreground text-center py-4">
                No conversations yet
              </p>
            ) : (
              conversations.map((c) => (
                <div
                  key={c.id}
                  className={`group flex items-center justify-between rounded-md px-3 py-2 text-sm cursor-pointer transition-colors ${
                    activeConvoId === c.id
                      ? "bg-accent text-accent-foreground"
                      : "hover:bg-accent/50 text-muted-foreground"
                  }`}
                  onClick={() => setActiveConvoId(c.id)}
                >
                  <div className="flex items-center gap-2 min-w-0">
                    <MessageSquare className="h-4 w-4 shrink-0" />
                    <span className="truncate">{c.title || "Untitled"}</span>
                  </div>
                  <button
                    onClick={(e) => {
                      e.stopPropagation();
                      deleteConversation(c.id);
                    }}
                    className="opacity-0 group-hover:opacity-100 text-muted-foreground hover:text-destructive"
                  >
                    <Trash2 className="h-3 w-3" />
                  </button>
                </div>
              ))
            )}
          </div>
        </ScrollArea>
      </Card>

      {/* Chat area */}
      <Card className="flex-1 flex flex-col">
        {/* Messages */}
        <ScrollArea className="flex-1 p-4">
          {messages.length === 0 ? (
            <div className="flex flex-col items-center justify-center h-full text-center">
              <MessageSquare className="h-12 w-12 text-muted-foreground mb-4" />
              <p className="text-lg font-medium">Ask a question about your documents</p>
              <p className="text-sm text-muted-foreground mt-1">
                I can help you find information, summarize content, and answer questions.
              </p>
            </div>
          ) : (
            <div className="space-y-4 max-w-3xl mx-auto">
              {messages.map((msg) => (
                <div
                  key={msg.id}
                  className={`flex ${msg.role === "user" ? "justify-end" : "justify-start"}`}
                >
                  <div
                    className={`max-w-[80%] rounded-lg px-4 py-3 ${
                      msg.role === "user"
                        ? "bg-primary text-primary-foreground"
                        : "bg-muted"
                    }`}
                  >
                    <p className="text-sm whitespace-pre-wrap">{msg.content}</p>

                    {/* Sources */}
                    {msg.sources && msg.sources.length > 0 && (
                      <div className="mt-3 pt-2 border-t border-border/50">
                        <p className="text-xs font-medium mb-1 opacity-70">Sources:</p>
                        <div className="flex flex-wrap gap-1">
                          {msg.sources.map((s, i) => (
                            <Badge key={i} variant="outline" className="text-xs">
                              <FileText className="h-3 w-3 mr-1" />
                              {s.document_name || `Chunk ${s.chunk_id}`}
                            </Badge>
                          ))}
                        </div>
                      </div>
                    )}

                    {/* Loading indicator for streaming */}
                    {msg.role === "assistant" && !msg.content && sending && (
                      <div className="flex items-center gap-2">
                        <Loader2 className="h-4 w-4 animate-spin" />
                        <span className="text-xs">Thinking...</span>
                      </div>
                    )}
                  </div>
                </div>
              ))}
              <div ref={messagesEndRef} />
            </div>
          )}
        </ScrollArea>

        <Separator />

        {/* Input */}
        <CardContent className="p-4">
          <div className="flex gap-2 max-w-3xl mx-auto">
            <Textarea
              value={input}
              onChange={(e) => setInput(e.target.value)}
              onKeyDown={handleKeyDown}
              placeholder="Ask a question about your documents..."
              className="min-h-[44px] max-h-[120px] resize-none"
              rows={1}
            />
            <Button
              onClick={handleSend}
              disabled={!input.trim() || sending}
              size="icon"
              className="shrink-0"
            >
              {isStreaming ? (
                <Loader2 className="h-4 w-4 animate-spin" />
              ) : (
                <Send className="h-4 w-4" />
              )}
            </Button>
          </div>
        </CardContent>
      </Card>
    </div>
  );
}
