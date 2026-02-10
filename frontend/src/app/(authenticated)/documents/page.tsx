"use client";

import { useEffect, useState, useCallback } from "react";
import Link from "next/link";
import { FileText, RefreshCw, Plus } from "lucide-react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Skeleton } from "@/components/ui/skeleton";
import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
  DialogTrigger,
} from "@/components/ui/dialog";
import { UploadZone } from "@/components/documents/upload-zone";
import { api } from "@/lib/api";
import { formatDate, formatFileSize } from "@/lib/utils";
import type { Document } from "@/types";

const statusVariant: Record<string, "default" | "secondary" | "destructive" | "outline"> = {
  uploaded: "outline",
  processing: "secondary",
  processed: "secondary",
  classified: "default",
  error: "destructive",
};

export default function DocumentsPage() {
  const [documents, setDocuments] = useState<Document[]>([]);
  const [total, setTotal] = useState(0);
  const [loading, setLoading] = useState(true);
  const [uploadOpen, setUploadOpen] = useState(false);

  const loadDocuments = useCallback(async () => {
    setLoading(true);
    try {
      const res = await api.get<{ documents: Document[]; total: number }>(
        "/documents/?limit=100"
      );
      setDocuments(res.documents || []);
      setTotal(res.total || 0);
    } catch {
      // API unavailable
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    loadDocuments();
  }, [loadDocuments]);

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-3xl font-bold">Documents</h1>
          <p className="text-muted-foreground">{total} documents total</p>
        </div>
        <div className="flex gap-2">
          <Button variant="outline" size="icon" onClick={loadDocuments}>
            <RefreshCw className="h-4 w-4" />
          </Button>
          <Dialog open={uploadOpen} onOpenChange={setUploadOpen}>
            <DialogTrigger asChild>
              <Button>
                <Plus className="mr-2 h-4 w-4" />
                Upload
              </Button>
            </DialogTrigger>
            <DialogContent className="max-w-lg">
              <DialogHeader>
                <DialogTitle>Upload Documents</DialogTitle>
              </DialogHeader>
              <UploadZone
                onUploadComplete={() => {
                  setUploadOpen(false);
                  loadDocuments();
                }}
              />
            </DialogContent>
          </Dialog>
        </div>
      </div>

      {loading ? (
        <div className="space-y-3">
          {[...Array(5)].map((_, i) => (
            <Skeleton key={i} className="h-16 w-full" />
          ))}
        </div>
      ) : documents.length === 0 ? (
        <Card>
          <CardContent className="flex flex-col items-center justify-center py-12">
            <FileText className="h-12 w-12 text-muted-foreground mb-4" />
            <p className="text-lg font-medium mb-1">No documents yet</p>
            <p className="text-sm text-muted-foreground mb-4">
              Upload your first document to get started
            </p>
            <Button onClick={() => setUploadOpen(true)}>
              <Plus className="mr-2 h-4 w-4" />
              Upload Document
            </Button>
          </CardContent>
        </Card>
      ) : (
        <Card>
          <CardHeader>
            <CardTitle>All Documents</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="overflow-x-auto">
              <table className="w-full text-sm">
                <thead>
                  <tr className="border-b text-left">
                    <th className="pb-3 font-medium text-muted-foreground">Name</th>
                    <th className="pb-3 font-medium text-muted-foreground">Type</th>
                    <th className="pb-3 font-medium text-muted-foreground">Size</th>
                    <th className="pb-3 font-medium text-muted-foreground">Status</th>
                    <th className="pb-3 font-medium text-muted-foreground">Chunks</th>
                    <th className="pb-3 font-medium text-muted-foreground">Date</th>
                  </tr>
                </thead>
                <tbody>
                  {documents.map((doc) => (
                    <tr key={doc.id} className="border-b last:border-0 hover:bg-accent/50">
                      <td className="py-3">
                        <Link
                          href={`/documents/${doc.id}`}
                          className="flex items-center gap-2 font-medium text-primary hover:underline"
                        >
                          <FileText className="h-4 w-4 shrink-0" />
                          <span className="truncate max-w-[200px]">{doc.filename}</span>
                        </Link>
                      </td>
                      <td className="py-3 text-muted-foreground">
                        {doc.mime_type?.split("/").pop()?.toUpperCase() || "-"}
                      </td>
                      <td className="py-3 text-muted-foreground">
                        {doc.file_size ? formatFileSize(doc.file_size) : "-"}
                      </td>
                      <td className="py-3">
                        <Badge variant={statusVariant[doc.status] || "outline"}>
                          {doc.status}
                        </Badge>
                      </td>
                      <td className="py-3 text-muted-foreground">{doc.chunk_count || 0}</td>
                      <td className="py-3 text-muted-foreground">{formatDate(doc.created_at)}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </CardContent>
        </Card>
      )}
    </div>
  );
}
