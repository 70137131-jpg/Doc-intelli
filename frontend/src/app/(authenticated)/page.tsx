"use client";

import { useEffect, useState } from "react";
import Link from "next/link";
import {
  FileText,
  Upload,
  MessageSquare,
  Search,
  TrendingUp,
  Clock,
} from "lucide-react";
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Skeleton } from "@/components/ui/skeleton";
import { api } from "@/lib/api";
import { formatDate } from "@/lib/utils";
import type { Document } from "@/types";
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
} from "recharts";

interface DashboardStats {
  total_documents: number;
  processing: number;
  classified: number;
  total_chunks: number;
}

const categoryColors: Record<string, string> = {
  Invoice: "bg-blue-500",
  Resume: "bg-green-500",
  Contract: "bg-purple-500",
  Report: "bg-orange-500",
  Letter: "bg-pink-500",
  Other: "bg-gray-500",
};

export default function DashboardPage() {
  const [stats, setStats] = useState<DashboardStats | null>(null);
  const [recentDocs, setRecentDocs] = useState<Document[]>([]);
  const [chartData, setChartData] = useState<{ name: string; count: number }[]>([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    async function load() {
      try {
        const [docsRes, classRes] = await Promise.all([
          api.get<{ documents: Document[]; total: number }>("/documents/?limit=50"),
          api.get<{ classifications: { category: string }[] }>("/classification/"),
        ]);

        const docs = docsRes.documents || [];
        const classifications = classRes.classifications || [];

        // Compute stats
        const total = docsRes.total || docs.length;
        const processing = docs.filter((d) => d.status === "processing").length;
        const classified = docs.filter((d) => d.status === "classified").length;
        const totalChunks = docs.reduce((sum, d) => sum + (d.chunk_count || 0), 0);

        setStats({ total_documents: total, processing, classified, total_chunks: totalChunks });
        setRecentDocs(docs.slice(0, 5));

        // Category chart
        const catCount: Record<string, number> = {};
        classifications.forEach((c) => {
          catCount[c.category] = (catCount[c.category] || 0) + 1;
        });
        setChartData(
          Object.entries(catCount).map(([name, count]) => ({ name, count }))
        );
      } catch {
        // API not available — show empty state
      } finally {
        setLoading(false);
      }
    }
    load();
  }, []);

  if (loading) {
    return (
      <div className="space-y-6">
        <h1 className="text-3xl font-bold">Dashboard</h1>
        <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-4">
          {[...Array(4)].map((_, i) => (
            <Card key={i}>
              <CardHeader className="pb-2">
                <Skeleton className="h-4 w-24" />
              </CardHeader>
              <CardContent>
                <Skeleton className="h-8 w-16" />
              </CardContent>
            </Card>
          ))}
        </div>
        <Skeleton className="h-64 w-full" />
      </div>
    );
  }

  const statCards = [
    {
      label: "Total Documents",
      value: stats?.total_documents ?? 0,
      icon: FileText,
      color: "text-blue-500",
    },
    {
      label: "Processing",
      value: stats?.processing ?? 0,
      icon: Clock,
      color: "text-yellow-500",
    },
    {
      label: "Classified",
      value: stats?.classified ?? 0,
      icon: TrendingUp,
      color: "text-green-500",
    },
    {
      label: "Total Chunks",
      value: stats?.total_chunks ?? 0,
      icon: Search,
      color: "text-purple-500",
    },
  ];

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <h1 className="text-3xl font-bold">Dashboard</h1>
        <Link href="/documents">
          <Button>
            <Upload className="mr-2 h-4 w-4" />
            Upload Document
          </Button>
        </Link>
      </div>

      {/* Stat cards */}
      <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-4">
        {statCards.map((s) => (
          <Card key={s.label}>
            <CardHeader className="flex flex-row items-center justify-between pb-2">
              <CardTitle className="text-sm font-medium text-muted-foreground">
                {s.label}
              </CardTitle>
              <s.icon className={`h-5 w-5 ${s.color}`} />
            </CardHeader>
            <CardContent>
              <div className="text-2xl font-bold">{s.value}</div>
            </CardContent>
          </Card>
        ))}
      </div>

      <div className="grid gap-6 lg:grid-cols-2">
        {/* Classification chart */}
        <Card>
          <CardHeader>
            <CardTitle>Documents by Category</CardTitle>
            <CardDescription>Classification distribution</CardDescription>
          </CardHeader>
          <CardContent>
            {chartData.length > 0 ? (
              <ResponsiveContainer width="100%" height={250}>
                <BarChart data={chartData}>
                  <CartesianGrid strokeDasharray="3 3" className="stroke-muted" />
                  <XAxis dataKey="name" className="text-xs" />
                  <YAxis className="text-xs" />
                  <Tooltip />
                  <Bar dataKey="count" fill="hsl(var(--primary))" radius={[4, 4, 0, 0]} />
                </BarChart>
              </ResponsiveContainer>
            ) : (
              <div className="flex h-[250px] items-center justify-center text-muted-foreground">
                No classifications yet
              </div>
            )}
          </CardContent>
        </Card>

        {/* Recent documents */}
        <Card>
          <CardHeader>
            <CardTitle>Recent Documents</CardTitle>
            <CardDescription>Latest uploaded documents</CardDescription>
          </CardHeader>
          <CardContent>
            {recentDocs.length > 0 ? (
              <div className="space-y-3">
                {recentDocs.map((doc) => (
                  <Link
                    key={doc.id}
                    href={`/documents/${doc.id}`}
                    className="flex items-center justify-between rounded-md border p-3 hover:bg-accent transition-colors"
                  >
                    <div className="flex items-center gap-3 min-w-0">
                      <FileText className="h-5 w-5 text-muted-foreground shrink-0" />
                      <div className="min-w-0">
                        <p className="text-sm font-medium truncate">{doc.filename}</p>
                        <p className="text-xs text-muted-foreground">
                          {formatDate(doc.created_at)}
                        </p>
                      </div>
                    </div>
                    <Badge
                      variant={doc.status === "classified" ? "default" : "secondary"}
                    >
                      {doc.status}
                    </Badge>
                  </Link>
                ))}
              </div>
            ) : (
              <div className="flex h-[250px] items-center justify-center text-muted-foreground">
                No documents yet
              </div>
            )}
          </CardContent>
        </Card>
      </div>

      {/* Quick actions */}
      <Card>
        <CardHeader>
          <CardTitle>Quick Actions</CardTitle>
        </CardHeader>
        <CardContent className="flex flex-wrap gap-3">
          <Link href="/documents">
            <Button variant="outline">
              <FileText className="mr-2 h-4 w-4" />
              Browse Documents
            </Button>
          </Link>
          <Link href="/chat">
            <Button variant="outline">
              <MessageSquare className="mr-2 h-4 w-4" />
              Ask a Question
            </Button>
          </Link>
          <Link href="/search">
            <Button variant="outline">
              <Search className="mr-2 h-4 w-4" />
              Search Documents
            </Button>
          </Link>
        </CardContent>
      </Card>
    </div>
  );
}
