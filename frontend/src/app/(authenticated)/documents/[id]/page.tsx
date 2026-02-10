"use client";

import { useEffect, useState } from "react";
import { useParams, useRouter } from "next/navigation";
import { ArrowLeft, FileText, Tag, Layers } from "lucide-react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Skeleton } from "@/components/ui/skeleton";
import { Separator } from "@/components/ui/separator";
import { api } from "@/lib/api";
import { formatDate, formatFileSize } from "@/lib/utils";
import type { Document, DocumentChunk, Classification, ExtractedField } from "@/types";

export default function DocumentDetailPage() {
  const params = useParams();
  const router = useRouter();
  const id = params.id as string;

  const [doc, setDoc] = useState<Document | null>(null);
  const [chunks, setChunks] = useState<DocumentChunk[]>([]);
  const [classification, setClassification] = useState<Classification | null>(null);
  const [fields, setFields] = useState<ExtractedField[]>([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    async function load() {
      try {
        const [docRes, chunksRes] = await Promise.all([
          api.get<Document>(`/documents/${id}`),
          api.get<{ chunks: DocumentChunk[] }>(`/documents/${id}/chunks`).catch(() => ({ chunks: [] })),
        ]);

        setDoc(docRes);
        setChunks(chunksRes.chunks || []);

        // Try loading classification
        try {
          const classRes = await api.get<{ classifications: Classification[] }>(
            `/classification/?document_id=${id}`
          );
          if (classRes.classifications?.length) {
            setClassification(classRes.classifications[0]);
          }
        } catch {
          // No classification yet
        }

        // Try loading extracted fields
        try {
          const fieldsRes = await api.get<{ fields: ExtractedField[] }>(
            `/documents/${id}/fields`
          );
          setFields(fieldsRes.fields || []);
        } catch {
          // No fields
        }
      } catch {
        // Document not found
      } finally {
        setLoading(false);
      }
    }
    load();
  }, [id]);

  if (loading) {
    return (
      <div className="space-y-6">
        <Skeleton className="h-8 w-48" />
        <Skeleton className="h-64 w-full" />
      </div>
    );
  }

  if (!doc) {
    return (
      <div className="flex flex-col items-center justify-center py-12">
        <p className="text-lg font-medium mb-4">Document not found</p>
        <Button variant="outline" onClick={() => router.push("/documents")}>
          <ArrowLeft className="mr-2 h-4 w-4" />
          Back to Documents
        </Button>
      </div>
    );
  }

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center gap-4">
        <Button variant="ghost" size="icon" onClick={() => router.push("/documents")}>
          <ArrowLeft className="h-5 w-5" />
        </Button>
        <div>
          <h1 className="text-2xl font-bold flex items-center gap-2">
            <FileText className="h-6 w-6" />
            {doc.filename}
          </h1>
          <p className="text-muted-foreground text-sm">
            Uploaded {formatDate(doc.created_at)}
            {doc.file_size && ` · ${formatFileSize(doc.file_size)}`}
          </p>
        </div>
        <Badge className="ml-auto" variant={doc.status === "classified" ? "default" : "secondary"}>
          {doc.status}
        </Badge>
      </div>

      <div className="grid gap-6 lg:grid-cols-3">
        {/* Document info */}
        <Card className="lg:col-span-1">
          <CardHeader>
            <CardTitle className="text-lg">Details</CardTitle>
          </CardHeader>
          <CardContent className="space-y-3 text-sm">
            <div className="flex justify-between">
              <span className="text-muted-foreground">Type</span>
              <span>{doc.mime_type?.split("/").pop()?.toUpperCase() || "-"}</span>
            </div>
            <Separator />
            <div className="flex justify-between">
              <span className="text-muted-foreground">Size</span>
              <span>{doc.file_size ? formatFileSize(doc.file_size) : "-"}</span>
            </div>
            <Separator />
            <div className="flex justify-between">
              <span className="text-muted-foreground">Chunks</span>
              <span>{doc.chunk_count || chunks.length}</span>
            </div>
            <Separator />
            <div className="flex justify-between">
              <span className="text-muted-foreground">Status</span>
              <Badge variant="outline" className="text-xs">
                {doc.status}
              </Badge>
            </div>

            {classification && (
              <>
                <Separator />
                <div className="flex justify-between items-center">
                  <span className="text-muted-foreground flex items-center gap-1">
                    <Tag className="h-3 w-3" />
                    Category
                  </span>
                  <Badge>{classification.category}</Badge>
                </div>
                <div className="flex justify-between">
                  <span className="text-muted-foreground">Confidence</span>
                  <span>{(classification.confidence * 100).toFixed(1)}%</span>
                </div>
              </>
            )}
          </CardContent>
        </Card>

        {/* Content area */}
        <div className="lg:col-span-2 space-y-6">
          {/* Extracted fields */}
          {fields.length > 0 && (
            <Card>
              <CardHeader>
                <CardTitle className="text-lg flex items-center gap-2">
                  <Tag className="h-5 w-5" />
                  Extracted Fields
                </CardTitle>
              </CardHeader>
              <CardContent>
                <div className="grid gap-2 sm:grid-cols-2">
                  {fields.map((f, i) => (
                    <div key={i} className="rounded-md border p-3">
                      <p className="text-xs text-muted-foreground font-medium uppercase">
                        {f.field_name}
                      </p>
                      <p className="text-sm mt-0.5">{f.field_value}</p>
                    </div>
                  ))}
                </div>
              </CardContent>
            </Card>
          )}

          {/* Chunks */}
          <Card>
            <CardHeader>
              <CardTitle className="text-lg flex items-center gap-2">
                <Layers className="h-5 w-5" />
                Document Chunks ({chunks.length})
              </CardTitle>
            </CardHeader>
            <CardContent>
              {chunks.length > 0 ? (
                <div className="space-y-3">
                  {chunks.map((chunk, i) => (
                    <div key={chunk.id || i} className="rounded-md border p-4">
                      <div className="flex items-center justify-between mb-2">
                        <Badge variant="outline" className="text-xs">
                          Chunk {chunk.chunk_index + 1}
                        </Badge>
                        <span className="text-xs text-muted-foreground">
                          {chunk.content.length} chars
                        </span>
                      </div>
                      <p className="text-sm whitespace-pre-wrap leading-relaxed">
                        {chunk.content}
                      </p>
                    </div>
                  ))}
                </div>
              ) : (
                <p className="text-sm text-muted-foreground py-4 text-center">
                  No chunks available. Document may still be processing.
                </p>
              )}
            </CardContent>
          </Card>
        </div>
      </div>
    </div>
  );
}
