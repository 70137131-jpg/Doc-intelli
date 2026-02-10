export interface User {
  id: string;
  email: string;
  full_name: string | null;
  role: "admin" | "user" | "viewer";
  is_active: boolean;
  created_at: string;
}

export interface Document {
  id: string;
  filename: string;
  original_filename?: string;
  mime_type: string;
  file_size?: number;
  file_size_bytes?: number;
  status: string;
  page_count?: number | null;
  chunk_count?: number | null;
  total_chunks?: number | null;
  error_message?: string | null;
  created_at: string;
  updated_at?: string;
}

export interface DocumentChunk {
  id: string;
  document_id: string;
  page_number: number | null;
  chunk_index: number;
  content: string;
  token_count: number;
  section_header: string | null;
}

export interface Classification {
  id: string;
  document_id: string;
  category: string;
  document_type?: string;
  confidence: number;
  method?: string;
  reasoning?: string | null;
}

export interface ExtractedField {
  id: string;
  field_name: string;
  field_value: string;
  field_type: string;
  confidence: number;
  extraction_method: string;
}

export interface SearchResult {
  chunk_id?: string;
  document_id: string;
  document_name?: string;
  content: string;
  score?: number;
  category?: string;
  chunk_index?: number;
  page_number?: number | null;
  section_header?: string | null;
}

export interface Conversation {
  id: string;
  title: string;
  created_at: string;
  updated_at: string;
}

export interface Message {
  id: string;
  role: "user" | "assistant";
  content: string;
  sources?: Source[];
  confidence?: string;
  created_at: string;
}

export interface Source {
  chunk_id: string;
  document_name: string;
  page_number: number | null;
  score: number;
}

export interface AgentWorkflowParam {
  name: string;
  type: string;
  label: string;
  required: boolean;
}

export interface AgentWorkflow {
  id: string;
  name: string;
  description: string;
  icon: string;
  parameters: AgentWorkflowParam[];
  required_params?: string[];
}

export interface TokenResponse {
  access_token: string;
  refresh_token: string;
  token_type: string;
}
