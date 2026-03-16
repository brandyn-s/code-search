"""Tests for HCL/Terraform chunker."""

from chunking.languages.hcl_chunker import HclChunker


def test_hcl_chunker_splits_on_blocks():
    """HCL chunker should produce chunks for resource/variable/output blocks."""
    chunker = HclChunker()
    source = """variable "vpc_cidr" {
  type    = string
  default = "10.0.0.0/16"
}

resource "aws_vpc" "main" {
  cidr_block = var.vpc_cidr
  tags = {
    Name = "main"
  }
}

output "vpc_id" {
  value = aws_vpc.main.id
}
"""
    chunks = chunker.chunk_code(source)
    names = [c.metadata.get("name", "") for c in chunks]

    assert len(chunks) == 3
    assert any("vpc_cidr" in n for n in names)
    assert any("aws_vpc" in n for n in names)
    assert any("vpc_id" in n for n in names)
