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


def test_nix_chunker_detects_mkOption_pattern():
    """mkOption bindings should have nix_pattern and is_option_declaration metadata."""
    chunker = NixChunker()
    source = """
{ config, lib, ... }:
{
  options.myservice.ip = lib.mkOption {
    type = lib.types.str;
    default = "192.168.1.1";
    description = "IP address";
  };
}
"""
    chunks = chunker.chunk_code(source)
    ip_chunks = [c for c in chunks if c.metadata.get("name", "").endswith("ip")]
    assert len(ip_chunks) >= 1, f"No ip chunk found in {[c.metadata.get('name') for c in chunks]}"
    ip = ip_chunks[0]
    assert ip.metadata.get("nix_pattern") == "mkOption"
    assert ip.metadata.get("is_option_declaration") is True


def test_nix_chunker_detects_mkEnableOption_pattern():
    """mkEnableOption bindings should be detected as option declarations."""
    chunker = NixChunker()
    # Binding must be >= 5 lines (MIN_CHUNK_LINES) to produce a chunk
    source = """
{ config, lib, ... }:
{
  options.myservice = {
    enable = lib.mkEnableOption ''
      my custom service for
      doing important things
      on the boat network
    '';
  };
}
"""
    chunks = chunker.chunk_code(source)
    enable_chunks = [c for c in chunks if "enable" in c.metadata.get("name", "")]
    assert len(enable_chunks) >= 1, f"No enable chunk found in {[c.metadata.get('name') for c in chunks]}"
    en = enable_chunks[0]
    assert en.metadata.get("nix_pattern") == "mkEnableOption"
    assert en.metadata.get("is_option_declaration") is True


def test_nix_chunker_detects_mkIf_pattern():
    """mkIf bindings should be detected as conditional."""
    chunker = NixChunker()
    source = """
{ config, lib, ... }:
let cfg = config.myservice; in
{
  config = lib.mkIf cfg.enable {
    networking.firewall.allowedTCPPorts = [ 80 443 ];
    systemd.services.myapp = {
      wantedBy = [ "multi-user.target" ];
    };
  };
}
"""
    chunks = chunker.chunk_code(source)
    config_chunks = [c for c in chunks if c.metadata.get("name") == "config"]
    # config may or may not appear depending on child coverage, check any chunk
    mkif_chunks = [c for c in chunks if c.metadata.get("nix_pattern") == "mkIf"]
    assert len(mkif_chunks) >= 1 or any(
        c.metadata.get("is_conditional") for c in chunks
    ), f"No mkIf pattern detected in {[c.metadata for c in chunks]}"


def test_nix_chunker_detects_service_category():
    """Bindings under services.* should get nix_category=service."""
    chunker = NixChunker()
    source = """
{ config, lib, ... }:
{
  services.myapp = {
    enable = true;
    package = pkgs.myapp;
    extraConfig = ''
      listen 0.0.0.0:8080;
    '';
  };
}
"""
    chunks = chunker.chunk_code(source)
    svc_chunks = [c for c in chunks if c.metadata.get("nix_category") == "service"]
    assert len(svc_chunks) >= 1, (
        f"No service category found in {[c.metadata for c in chunks]}"
    )


def test_nix_chunker_detects_networking_category():
    """Bindings under networking.* should get nix_category=networking."""
    chunker = NixChunker()
    source = """
{ config, lib, ... }:
{
  networking.firewall = {
    enable = true;
    allowedTCPPorts = [ 80 443 8080 ];
    allowedUDPPorts = [ 53 ];
  };
}
"""
    chunks = chunker.chunk_code(source)
    net_chunks = [c for c in chunks if c.metadata.get("nix_category") == "networking"]
    assert len(net_chunks) >= 1, (
        f"No networking category found in {[c.metadata for c in chunks]}"
    )


def test_nix_chunker_detects_imports_category():
    """imports = [...] bindings should get nix_category=imports."""
    chunker = NixChunker()
    source = """
{ config, lib, ... }:
{
  imports = [
    ./hardware.nix
    ./networking.nix
    ./services.nix
  ];
}
"""
    chunks = chunker.chunk_code(source)
    import_chunks = [c for c in chunks if c.metadata.get("nix_category") == "imports"]
    assert len(import_chunks) >= 1, (
        f"No imports category found in {[c.metadata for c in chunks]}"
    )


def test_nix_chunker_detects_option_declaration_category():
    """Bindings under options.* should get nix_category=option_declaration."""
    chunker = NixChunker()
    source = """
{ config, lib, ... }:
{
  options.myservice = {
    enable = lib.mkEnableOption "my service";
    port = lib.mkOption {
      type = lib.types.port;
      default = 8080;
    };
  };
}
"""
    chunks = chunker.chunk_code(source)
    opt_chunks = [c for c in chunks if c.metadata.get("nix_category") == "option_declaration"]
    assert len(opt_chunks) >= 1, (
        f"No option_declaration category found in {[c.metadata for c in chunks]}"
    )
