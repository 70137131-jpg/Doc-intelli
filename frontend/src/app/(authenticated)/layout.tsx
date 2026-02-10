"use client";

import { AppShell } from "@/components/layout/app-shell";
import { ErrorBoundary } from "@/components/layout/error-boundary";
import { useKeyboardShortcuts } from "@/hooks/use-keyboard-shortcuts";

export default function AuthenticatedLayout({ children }: { children: React.ReactNode }) {
  useKeyboardShortcuts();

  return (
    <ErrorBoundary>
      <AppShell>{children}</AppShell>
    </ErrorBoundary>
  );
}
