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
    names = [c.metadata.get("name", "") for c in chunks]

    # Dedup skips parent bindings when children cover >50% of their lines,
    # so we get leaf-level chunks rather than top-level parents
    assert len(chunks) >= 2, f"Expected multiple chunks, got {len(chunks)}: {names}"
    assert any("ip" in n for n in names), f"No ip binding found in {names}"
    assert any("port" in n for n in names), f"No port binding found in {names}"
    assert any("systemd" in n for n in names), f"No systemd binding found in {names}"


def test_nix_chunker_skips_parent_when_children_cover_content():
    """Parent binding should not emit when children cover >50% of its lines."""
    chunker = NixChunker()
    source = """
{ config, lib, ... }:
{
  config = {
    networking = {
      firewall.enable = true;
      firewall.allowedTCPPorts = [ 80 443 ];
      interfaces.eth0 = {
        useDHCP = true;
      };
    };
    systemd.services.myapp = {
      description = "My App";
      wantedBy = [ "multi-user.target" ];
      serviceConfig = {
        ExecStart = "/bin/myapp";
        Restart = "always";
        Type = "simple";
      };
    };
  };
}
"""
    chunks = chunker.chunk_code(source)
    names = [c.metadata.get("name", "") for c in chunks]

    # Leaf children should be present (networking has no big children so it emits;
    # systemd.services.myapp gets deduped because serviceConfig covers >50%)
    assert any("networking" in n for n in names), f"No networking chunk in {names}"
    assert any("serviceConfig" in n for n in names), (
        f"No serviceConfig chunk in {names}"
    )

    # Parent 'config' should NOT be a separate chunk (children cover it)
    config_chunks = [c for c in chunks if c.metadata.get("name") == "config"]
    assert len(config_chunks) == 0, (
        f"Parent 'config' should be skipped, got {len(config_chunks)} chunks"
    )
    # Intermediate 'systemd.services.myapp' should also be skipped
    systemd_chunks = [
        c for c in chunks if c.metadata.get("name") == "systemd.services.myapp"
    ]
    assert len(systemd_chunks) == 0, (
        f"Intermediate 'systemd.services.myapp' should be skipped, got {len(systemd_chunks)} chunks"
    )


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
