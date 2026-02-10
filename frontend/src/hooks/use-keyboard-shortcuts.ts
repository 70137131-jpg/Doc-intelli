"use client";

import { useEffect } from "react";
import { useRouter } from "next/navigation";

export function useKeyboardShortcuts() {
  const router = useRouter();

  useEffect(() => {
    function handleKeyDown(e: KeyboardEvent) {
      const isMod = e.metaKey || e.ctrlKey;

      // Cmd/Ctrl + K → go to search
      if (isMod && e.key === "k") {
        e.preventDefault();
        router.push("/search");
      }

      // Cmd/Ctrl + N → new chat
      if (isMod && e.key === "n") {
        e.preventDefault();
        router.push("/chat");
      }

      // Cmd/Ctrl + U → upload (go to documents)
      if (isMod && e.key === "u") {
        e.preventDefault();
        router.push("/documents");
      }
    }

    window.addEventListener("keydown", handleKeyDown);
    return () => window.removeEventListener("keydown", handleKeyDown);
  }, [router]);
}
