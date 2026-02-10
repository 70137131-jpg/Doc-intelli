"use client";

import { useState, useEffect } from "react";
import {
  Bot,
  Play,
  Loader2,
  StopCircle,
  FileText,
  Search,
  Sparkles,
  CheckCircle,
  AlertCircle,
} from "lucide-react";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Badge } from "@/components/ui/badge";
import { ScrollArea } from "@/components/ui/scroll-area";
import { Separator } from "@/components/ui/separator";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { useSSE } from "@/hooks/use-sse";
import { api } from "@/lib/api";
import type { AgentWorkflow } from "@/types";

interface ExecutionEvent {
  type: string;
  node?: string;
  content?: string;
  timestamp: string;
}

const workflowTemplates: AgentWorkflow[] = [
  {
    id: "document_analysis",
    name: "Document Analysis",
    description: "Analyze a document: extract text, classify, extract fields, and generate summary",
    icon: "FileText",
    parameters: [
      { name: "document_id", type: "string", label: "Document ID", required: true },
    ],
  },
  {
    id: "research_query",
    name: "Research Query",
    description: "Multi-step research: search documents, gather context, synthesize an answer",
    icon: "Search",
    parameters: [
      { name: "query", type: "string", label: "Research Question", required: true },
      { name: "max_sources", type: "number", label: "Max Sources", required: false },
    ],
  },
  {
    id: "batch_classify",
    name: "Batch Classification",
    description: "Classify multiple documents using the ML pipeline with confidence thresholds",
    icon: "Sparkles",
    parameters: [
      { name: "confidence_threshold", type: "number", label: "Min Confidence (%)", required: false },
    ],
  },
];

const iconMap: Record<string, React.ElementType> = {
  FileText,
  Search,
  Sparkles,
};

export default function AgentsPage() {
  const [selectedWorkflow, setSelectedWorkflow] = useState<AgentWorkflow | null>(null);
  const [params, setParams] = useState<Record<string, string>>({});
  const [events, setEvents] = useState<ExecutionEvent[]>([]);
  const [isRunning, setIsRunning] = useState(false);
  const { isStreaming, startStream, stopStream } = useSSE();

  const handleSelectWorkflow = (wf: AgentWorkflow) => {
    setSelectedWorkflow(wf);
    setParams({});
    setEvents([]);
  };

  const handleRun = async () => {
    if (!selectedWorkflow) return;
    setIsRunning(true);
    setEvents([]);

    try {
      await startStream(
        `/agents/execute/${selectedWorkflow.id}`,
        params,
        (event) => {
          const newEvent: ExecutionEvent = {
            type: event.type,
            node: event.node,
            content: event.data,
            timestamp: new Date().toISOString(),
          };
          setEvents((prev) => [...prev, newEvent]);
        }
      );
    } catch {
      setEvents((prev) => [
        ...prev,
        {
          type: "error",
          content: "Execution failed. Please try again.",
          timestamp: new Date().toISOString(),
        },
      ]);
    } finally {
      setIsRunning(false);
    }

    // Fallback for non-streaming API
    if (events.length === 0) {
      try {
        const res = await api.post<{ result: string; steps: ExecutionEvent[] }>(
          `/agents/execute/${selectedWorkflow.id}`,
          params
        );
        setEvents(
          res.steps || [
            { type: "result", content: res.result, timestamp: new Date().toISOString() },
          ]
        );
      } catch {
        setEvents([
          {
            type: "error",
            content: "Agent execution failed.",
            timestamp: new Date().toISOString(),
          },
        ]);
      }
    }
  };

  const handleStop = () => {
    stopStream();
    setIsRunning(false);
    setEvents((prev) => [
      ...prev,
      { type: "cancelled", content: "Execution stopped by user.", timestamp: new Date().toISOString() },
    ]);
  };

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-3xl font-bold">Agents</h1>
        <p className="text-muted-foreground">Run AI agent workflows on your documents</p>
      </div>

      {/* Workflow cards */}
      <div className="grid gap-4 md:grid-cols-3">
        {workflowTemplates.map((wf) => {
          const Icon = iconMap[wf.icon] || Bot;
          const isSelected = selectedWorkflow?.id === wf.id;
          return (
            <Card
              key={wf.id}
              className={`cursor-pointer transition-all hover:shadow-md ${
                isSelected ? "ring-2 ring-primary" : ""
              }`}
              onClick={() => handleSelectWorkflow(wf)}
            >
              <CardHeader>
                <div className="flex items-center gap-3">
                  <div className="rounded-lg bg-primary/10 p-2">
                    <Icon className="h-5 w-5 text-primary" />
                  </div>
                  <div>
                    <CardTitle className="text-base">{wf.name}</CardTitle>
                  </div>
                </div>
              </CardHeader>
              <CardContent>
                <CardDescription>{wf.description}</CardDescription>
              </CardContent>
            </Card>
          );
        })}
      </div>

      {/* Execution panel */}
      {selectedWorkflow && (
        <div className="grid gap-6 lg:grid-cols-2">
          {/* Parameters */}
          <Card>
            <CardHeader>
              <CardTitle className="text-lg">
                {selectedWorkflow.name} — Parameters
              </CardTitle>
            </CardHeader>
            <CardContent className="space-y-4">
              {selectedWorkflow.parameters.map((p) => (
                <div key={p.name} className="space-y-2">
                  <Label>
                    {p.label}
                    {p.required && <span className="text-destructive ml-1">*</span>}
                  </Label>
                  <Input
                    type={p.type === "number" ? "number" : "text"}
                    value={params[p.name] || ""}
                    onChange={(e) =>
                      setParams((prev) => ({ ...prev, [p.name]: e.target.value }))
                    }
                    placeholder={`Enter ${p.label.toLowerCase()}`}
                  />
                </div>
              ))}

              <div className="flex gap-2 pt-2">
                <Button onClick={handleRun} disabled={isRunning}>
                  {isRunning ? (
                    <>
                      <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                      Running...
                    </>
                  ) : (
                    <>
                      <Play className="mr-2 h-4 w-4" />
                      Execute
                    </>
                  )}
                </Button>
                {isRunning && (
                  <Button variant="destructive" onClick={handleStop}>
                    <StopCircle className="mr-2 h-4 w-4" />
                    Stop
                  </Button>
                )}
              </div>
            </CardContent>
          </Card>

          {/* Execution log */}
          <Card>
            <CardHeader>
              <CardTitle className="text-lg">Execution Log</CardTitle>
            </CardHeader>
            <CardContent>
              <ScrollArea className="h-[400px]">
                {events.length === 0 ? (
                  <div className="flex flex-col items-center justify-center h-full text-muted-foreground">
                    <Bot className="h-8 w-8 mb-2" />
                    <p className="text-sm">Run a workflow to see the execution log</p>
                  </div>
                ) : (
                  <div className="space-y-2 font-mono text-xs">
                    {events.map((ev, i) => (
                      <div
                        key={i}
                        className={`flex gap-2 items-start p-2 rounded ${
                          ev.type === "error"
                            ? "bg-destructive/10 text-destructive"
                            : ev.type === "result" || ev.type === "done"
                            ? "bg-green-500/10 text-green-700 dark:text-green-400"
                            : "bg-muted"
                        }`}
                      >
                        {ev.type === "error" ? (
                          <AlertCircle className="h-3 w-3 mt-0.5 shrink-0" />
                        ) : ev.type === "result" || ev.type === "done" ? (
                          <CheckCircle className="h-3 w-3 mt-0.5 shrink-0" />
                        ) : (
                          <Bot className="h-3 w-3 mt-0.5 shrink-0" />
                        )}
                        <div className="min-w-0">
                          {ev.node && (
                            <Badge variant="outline" className="text-[10px] mb-1">
                              {ev.node}
                            </Badge>
                          )}
                          <p className="whitespace-pre-wrap break-words">{ev.content}</p>
                        </div>
                      </div>
                    ))}
                  </div>
                )}
              </ScrollArea>
            </CardContent>
          </Card>
        </div>
      )}
    </div>
  );
}
