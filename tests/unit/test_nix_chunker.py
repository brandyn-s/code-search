"""Tests for Nix chunker granularity."""
from chunking.languages.nix_chunker import NixChunker


def test_nix_chunker_splits_bindings_inside_let():
    """A Nix module with let ... in { bindings } should produce multiple chunks."""
    chunker = NixChunker()
    source = """
{ config, lib, ... }:
with lib;
let
  cfg = config.test;
  isEnabled = cfg.enable;
in
{
  options.test = {
    enable = mkEnableOption "test";
    ip = mkOption {
      type = types.str;
      default = "192.168.1.1";
      description = "IP address";
    };
    port = mkOption {
      type = types.port;
      default = 8080;
      description = "Port number";
    };
  };
  config = mkIf cfg.enable {
    networking.firewall.allowedTCPPorts = [ cfg.port ];
    systemd.services.test = {
      description = "Test service";
      wantedBy = [ "multi-user.target" ];
      serviceConfig = {
        ExecStart = "test --ip ${cfg.ip}";
        Type = "simple";
      };
    };
  };
}
"""
    chunks = chunker.chunk_code(source)
    names = [c.metadata.get('name', '') for c in chunks]

    assert len(chunks) > 2, f"Expected multiple chunks, got {len(chunks)}: {names}"
    assert any('options' in n for n in names), f"No options binding found in {names}"
    assert any('config' in n for n in names), f"No config binding found in {names}"


def test_nix_chunker_small_bindings_stay_grouped():
    """Small bindings (< 5 lines) should not be separate chunks."""
    chunker = NixChunker()
    source = """
{
  x = 1;
  y = 2;
  z = 3;
}
"""
    chunks = chunker.chunk_code(source)
    assert len(chunks) <= 2
