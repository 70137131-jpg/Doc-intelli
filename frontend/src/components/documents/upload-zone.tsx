"use client";

import { useCallback, useState } from "react";
import { Upload, File, X } from "lucide-react";
import { cn } from "@/lib/utils";
import { Button } from "@/components/ui/button";
import { Progress } from "@/components/ui/progress";
import { api } from "@/lib/api";
import { useToast } from "@/components/ui/toast";
import { formatFileSize } from "@/lib/utils";

interface UploadZoneProps {
  onUploadComplete: () => void;
}

interface UploadFile {
  file: File;
  progress: number;
  status: "pending" | "uploading" | "done" | "error";
  error?: string;
}

const ACCEPTED_TYPES = [
  "application/pdf",
  "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
  "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
  "text/plain",
  "text/csv",
];

export function UploadZone({ onUploadComplete }: UploadZoneProps) {
  const { toast } = useToast();
  const [files, setFiles] = useState<UploadFile[]>([]);
  const [dragOver, setDragOver] = useState(false);

  const addFiles = useCallback((newFiles: FileList | File[]) => {
    const arr = Array.from(newFiles).map((file) => ({
      file,
      progress: 0,
      status: "pending" as const,
    }));
    setFiles((prev) => [...prev, ...arr]);
  }, []);

  const handleDrop = useCallback(
    (e: React.DragEvent) => {
      e.preventDefault();
      setDragOver(false);
      addFiles(e.dataTransfer.files);
    },
    [addFiles]
  );

  const handleFileInput = useCallback(
    (e: React.ChangeEvent<HTMLInputElement>) => {
      if (e.target.files) addFiles(e.target.files);
    },
    [addFiles]
  );

  const removeFile = (idx: number) => {
    setFiles((prev) => prev.filter((_, i) => i !== idx));
  };

  const uploadAll = async () => {
    for (let i = 0; i < files.length; i++) {
      if (files[i].status !== "pending") continue;

      setFiles((prev) => prev.map((f, j) => (j === i ? { ...f, status: "uploading" } : f)));

      try {
        await api.upload("/documents/upload", files[i].file);
        setFiles((prev) =>
          prev.map((f, j) => (j === i ? { ...f, status: "done", progress: 100 } : f))
        );
      } catch (err) {
        const message = err instanceof Error ? err.message : "Upload failed";
        setFiles((prev) =>
          prev.map((f, j) => (j === i ? { ...f, status: "error", error: message } : f))
        );
      }
    }
    toast({ title: "Upload complete", description: "Documents are being processed." });
    onUploadComplete();
  };

  const pendingCount = files.filter((f) => f.status === "pending").length;

  return (
    <div className="space-y-4">
      <div
        className={cn(
          "flex flex-col items-center justify-center rounded-lg border-2 border-dashed p-8 transition-colors",
          dragOver ? "border-primary bg-primary/5" : "border-muted-foreground/25"
        )}
        onDragOver={(e) => {
          e.preventDefault();
          setDragOver(true);
        }}
        onDragLeave={() => setDragOver(false)}
        onDrop={handleDrop}
      >
        <Upload className="h-10 w-10 text-muted-foreground mb-3" />
        <p className="text-sm text-muted-foreground mb-1">
          Drag & drop files here, or click to browse
        </p>
        <p className="text-xs text-muted-foreground mb-3">
          PDF, DOCX, XLSX, TXT, CSV (max 50 MB)
        </p>
        <input
          type="file"
          multiple
          accept={ACCEPTED_TYPES.join(",")}
          onChange={handleFileInput}
          className="hidden"
          id="file-upload"
        />
        <label htmlFor="file-upload">
          <Button variant="outline" asChild>
            <span>Browse Files</span>
          </Button>
        </label>
      </div>

      {files.length > 0 && (
        <div className="space-y-2">
          {files.map((f, i) => (
            <div key={i} className="flex items-center gap-3 rounded-md border p-3">
              <File className="h-5 w-5 text-muted-foreground shrink-0" />
              <div className="flex-1 min-w-0">
                <p className="text-sm font-medium truncate">{f.file.name}</p>
                <p className="text-xs text-muted-foreground">{formatFileSize(f.file.size)}</p>
                {f.status === "uploading" && <Progress value={50} className="mt-1 h-1" />}
                {f.status === "error" && (
                  <p className="text-xs text-destructive mt-1">{f.error}</p>
                )}
              </div>
              <span
                className={cn(
                  "text-xs font-medium",
                  f.status === "done" && "text-green-500",
                  f.status === "error" && "text-destructive",
                  f.status === "uploading" && "text-yellow-500"
                )}
              >
                {f.status === "done" ? "Done" : f.status === "error" ? "Failed" : ""}
              </span>
              {f.status === "pending" && (
                <button onClick={() => removeFile(i)} className="text-muted-foreground hover:text-foreground">
                  <X className="h-4 w-4" />
                </button>
              )}
            </div>
          ))}

          {pendingCount > 0 && (
            <Button onClick={uploadAll} className="w-full">
              <Upload className="mr-2 h-4 w-4" />
              Upload {pendingCount} file{pendingCount > 1 ? "s" : ""}
            </Button>
          )}
        </div>
      )}
    </div>
  );
}
