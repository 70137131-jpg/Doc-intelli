output "vpc_id" {
  description = "VPC ID"
  value       = aws_vpc.main.id
}

output "alb_dns_name" {
  description = "ALB DNS name"
  value       = aws_lb.main.dns_name
}

output "backend_ecr_url" {
  description = "Backend ECR repository URL"
  value       = aws_ecr_repository.backend.repository_url
}

output "frontend_ecr_url" {
  description = "Frontend ECR repository URL"
  value       = aws_ecr_repository.frontend.repository_url
}

output "rds_endpoint" {
  description = "RDS endpoint"
  value       = aws_db_instance.main.endpoint
}

output "redis_endpoint" {
  description = "ElastiCache Redis endpoint"
  value       = aws_elasticache_replication_group.main.primary_endpoint_address
}

output "s3_raw_bucket" {
  description = "S3 raw documents bucket"
  value       = aws_s3_bucket.raw_documents.id
}

output "s3_processed_bucket" {
  description = "S3 processed documents bucket"
  value       = aws_s3_bucket.processed_documents.id
}

output "ecs_cluster_name" {
  description = "ECS cluster name"
  value       = aws_ecs_cluster.main.name
}

output "sns_alerts_topic_arn" {
  description = "SNS alerts topic ARN"
  value       = aws_sns_topic.alerts.arn
}
