variable "aws_region" {
  description = "AWS region"
  type        = string
  default     = "us-east-1"
}

variable "environment" {
  description = "Environment name (staging/prod)"
  type        = string
  default     = "prod"
}

variable "project_name" {
  description = "Project name used for resource naming"
  type        = string
  default     = "docintelli"
}

variable "domain_name" {
  description = "Domain name for the application"
  type        = string
  default     = ""
}

# VPC
variable "vpc_cidr" {
  description = "VPC CIDR block"
  type        = string
  default     = "10.0.0.0/16"
}

# RDS
variable "db_instance_class" {
  description = "RDS instance class"
  type        = string
  default     = "db.t3.medium"
}

variable "db_name" {
  description = "Database name"
  type        = string
  default     = "docintelli"
}

variable "db_username" {
  description = "Database master username"
  type        = string
  default     = "docintelli"
  sensitive   = true
}

variable "db_password" {
  description = "Database master password"
  type        = string
  sensitive   = true
}

# ElastiCache
variable "redis_node_type" {
  description = "ElastiCache node type"
  type        = string
  default     = "cache.t3.micro"
}

# ECS
variable "backend_cpu" {
  description = "Backend task CPU units"
  type        = number
  default     = 1024
}

variable "backend_memory" {
  description = "Backend task memory (MB)"
  type        = number
  default     = 2048
}

variable "frontend_cpu" {
  description = "Frontend task CPU units"
  type        = number
  default     = 256
}

variable "frontend_memory" {
  description = "Frontend task memory (MB)"
  type        = number
  default     = 512
}

variable "celery_cpu" {
  description = "Celery worker CPU units"
  type        = number
  default     = 1024
}

variable "celery_memory" {
  description = "Celery worker memory (MB)"
  type        = number
  default     = 2048
}

variable "backend_desired_count" {
  description = "Desired number of backend tasks"
  type        = number
  default     = 2
}

variable "frontend_desired_count" {
  description = "Desired number of frontend tasks"
  type        = number
  default     = 2
}

variable "celery_desired_count" {
  description = "Desired number of Celery workers"
  type        = number
  default     = 2
}

# Secrets
variable "jwt_secret_key" {
  description = "JWT secret key"
  type        = string
  sensitive   = true
}

variable "gemini_api_key" {
  description = "Gemini API key"
  type        = string
  sensitive   = true
}
